import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Layout from "./Layout";
import OverviewPage from "./pages/OverviewPage";
import LaneExplorerPage from "./pages/LaneExplorerPage";
import ComparePage from "./pages/ComparePage";
import RunDetailPage from "./pages/RunDetailPage";
import PromotionBoardPage from "./pages/PromotionBoardPage";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<OverviewPage />} />
          <Route path="lanes" element={<LaneExplorerPage />} />
          <Route path="compare" element={<ComparePage />} />
          <Route path="runs/:summaryPath" element={<RunDetailPage />} />
          <Route path="promotions" element={<PromotionBoardPage />} />
          {/* Redirect unknown paths back to overview */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;

