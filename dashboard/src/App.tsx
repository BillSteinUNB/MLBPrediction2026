import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import Layout from "./Layout";
import LatestRunPage from "./pages/LatestRunPage";
import StageCardsPage from "./pages/StageCardsPage";
import BenchmarkComparePage from "./pages/BenchmarkComparePage";
import RunLedgerPage from "./pages/RunLedgerPage";
import BestRunsPage from "./pages/BestRunsPage";
import PromotionSummaryPage from "./pages/PromotionSummaryPage";
import SeasonPicksPage from "./pages/SeasonPicksPage";
import SeasonTrackingPage from "./pages/SeasonTrackingPage";
import SeasonSlatePage from "./pages/SeasonSlatePage";

export default function App() {
  return (
    <Router>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Navigate to="/season/picks" replace />} />
          <Route path="season/picks" element={<SeasonPicksPage />} />
          <Route path="season/tracking" element={<SeasonTrackingPage />} />
          <Route path="season/slate" element={<SeasonSlatePage />} />
          <Route path="research/latest" element={<LatestRunPage />} />
          <Route path="research/stages" element={<StageCardsPage />} />
          <Route path="research/benchmark" element={<BenchmarkComparePage />} />
          <Route path="history/ledger" element={<RunLedgerPage />} />
          <Route path="history/best" element={<BestRunsPage />} />
          <Route path="promotion" element={<PromotionSummaryPage />} />
          <Route path="*" element={<Navigate to="/season/picks" replace />} />
        </Route>
      </Routes>
    </Router>
  );
}
