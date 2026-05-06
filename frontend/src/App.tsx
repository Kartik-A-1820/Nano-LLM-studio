import { useState } from "react";

import { AppShell } from "./components/AppShell";
import { DataPage } from "./features/data/DataPage";
import { InferencePage } from "./features/inference/InferencePage";
import { TrainPage } from "./features/train/TrainPage";

type TabId = "data" | "train" | "inference";

export default function App() {
  const [currentTab, setCurrentTab] = useState<TabId>("data");

  return (
    <AppShell currentTab={currentTab} onTabChange={setCurrentTab}>
      {currentTab === "data" ? <DataPage /> : null}
      {currentTab === "train" ? <TrainPage /> : null}
      {currentTab === "inference" ? <InferencePage /> : null}
    </AppShell>
  );
}
