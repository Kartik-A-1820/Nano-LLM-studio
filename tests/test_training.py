from types import SimpleNamespace
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytest
from nano_llm.training.trainer import Trainer, TrainerConfig
from nano_llm.data.dataset import build_train_validation_loaders

class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.5))
        self.config = SimpleNamespace(name='toy')
    def forward(self, x, targets, use_checkpoint=False):
        return None, ((self.weight*x-targets)**2).mean()

def make(tmp_path, n=5, accumulation=4, epochs=1, validation=False, batch_size=1):
    data = TensorDataset(torch.arange(1,n+1,dtype=torch.float32).reshape(-1,1), torch.ones(n,1))
    loader = DataLoader(data,batch_size=batch_size,shuffle=False)
    cfg = TrainerConfig(epochs=epochs,grad_accum_steps=accumulation,use_fp16=False,use_gradient_checkpointing=False,checkpoint_dir=tmp_path,grad_clip=1000)
    return Trainer(ToyModel(),loader,cfg,None,device='cpu',validation_dataloader=loader if validation else None)

@pytest.mark.parametrize('n,accum,updates', [(5,4,2),(3,4,1),(1,1,1),(8,4,2)])
def test_all_partial_groups_are_updated(tmp_path,n,accum,updates):
    trainer=make(tmp_path,n,accum)
    trainer.train()
    assert trainer.global_step==updates
    assert trainer.scheduler.last_epoch==updates
    assert (tmp_path/'last.pt').is_file()
    assert not (tmp_path/'best.pt').exists()

def test_accumulated_gradient_matches_full_batch_with_short_microbatch(tmp_path):
    accumulated=make(tmp_path/'a',3,2,batch_size=2)
    full=make(tmp_path/'b',3,1,batch_size=3)
    for trainer in (accumulated,full):
        trainer.optimizer=torch.optim.SGD(trainer.model.parameters(),lr=.01)
        trainer.scheduler=torch.optim.lr_scheduler.LambdaLR(trainer.optimizer,lambda _:1.)
        trainer._train_epoch(1)
    assert torch.allclose(accumulated.model.weight,full.model.weight,atol=1e-7)

def test_best_checkpoint_is_selected_by_validation(tmp_path):
    trainer=make(tmp_path,epochs=3,validation=True)
    values=iter([.4,.6,.5])
    def train_epoch(epoch):
        with torch.no_grad():
            trainer.model.weight.fill_(epoch)
        return 1/epoch
    trainer._train_epoch=train_epoch
    trainer.evaluate=lambda:next(values)
    trainer.train()
    best=torch.load(tmp_path/'best.pt',weights_only=True)
    last=torch.load(tmp_path/'last.pt',weights_only=True)
    assert best['epoch']==1 and last['epoch']==3
    assert best['selection_metric']=='validation_loss'
    assert best['model_state']['weight']==1

def test_resume_restores_scheduler_and_epoch(tmp_path):
    uninterrupted=make(tmp_path/'full',epochs=4)
    uninterrupted.train()
    stopped=make(tmp_path/'resume',epochs=4)
    class StopAtTwo:
        def on_train_start(self,*a):pass
        def on_train_end(self,*a):pass
        def on_step_end(self,*a):pass
        def on_epoch_end(self,epoch,loss,trainer):
            if epoch==2:trainer.stop()
    stopped.callbacks=[StopAtTwo()]
    stopped.train()
    resumed=make(tmp_path/'resume',epochs=4)
    assert resumed.load_checkpoint(tmp_path/'resume'/'last.pt')==2
    resumed.train()
    assert resumed.global_step==uninterrupted.global_step
    assert resumed.scheduler.state_dict()==uninterrupted.scheduler.state_dict()
    assert torch.equal(resumed.model.weight,uninterrupted.model.weight)

def test_holdout_targets_do_not_overlap_and_small_batch_is_kept():
    data=TensorDataset(torch.arange(5).reshape(-1,1),torch.arange(5).reshape(-1,1))
    train,val=build_train_validation_loaders(data,10)
    assert len(train)==len(val)==1
    assert set(train.dataset.indices).isdisjoint(val.dataset.indices)
    assert sum(len(x) for x,_ in train)==4
    with pytest.raises(ValueError):
        build_train_validation_loaders(TensorDataset(torch.ones(1,1),torch.ones(1,1)),1)

def test_empty_loader_rejected(tmp_path):
    with pytest.raises(ValueError):
        make(tmp_path,n=0)

def test_real_transformer_train_validation_smoke(tmp_path):
    from nano_llm.model.config import ModelConfig
    from nano_llm.model.transformer import NanoLLM
    model=NanoLLM(ModelConfig(vocab_size=32,dim=32,n_layers=1,n_heads=4,n_kv_heads=2,max_seq_len=8,use_mla=False))
    tokens=torch.arange(24).reshape(3,8)%32
    loader=DataLoader(TensorDataset(tokens,(tokens+1)%32),batch_size=1)
    trainer=Trainer(model,loader,TrainerConfig(epochs=1,grad_accum_steps=4,use_fp16=False,use_gradient_checkpointing=False,checkpoint_dir=tmp_path),None,device='cpu',validation_dataloader=loader)
    trainer.train()
    assert trainer.global_step==1
    assert trainer.validation_loss>0
    assert (tmp_path/'best.pt').is_file()
