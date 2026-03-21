import React from "react";
import { NavLink, Outlet } from "react-router-dom";
import "./layout.css";

const Layout: React.FC = () => {
  return (
    <div className="mlb-layout">
      <aside className="mlb-sidebar" role="navigation" aria-label="Main">
        <div className="mlb-brand">MLB Experiment Dashboard</div>
        <nav>
          <ul className="mlb-nav">
            <li>
              <NavLink to="/" end className={({ isActive }) => (isActive ? "active" : "")}>
                Overview
              </NavLink>
            </li>
            <li>
              <NavLink to="/lanes" className={({ isActive }) => (isActive ? "active" : "")}>
                Lane Explorer
              </NavLink>
            </li>
            <li>
              <NavLink to="/compare" className={({ isActive }) => (isActive ? "active" : "")}>
                Compare
              </NavLink>
            </li>
            <li>
              <NavLink to="/promotions" className={({ isActive }) => (isActive ? "active" : "")}>
                Promotion Board
              </NavLink>
            </li>
            <li>
              <NavLink to="/runs/test" className={({ isActive }) => (isActive ? "active" : "")}>
                Run Detail (example)
              </NavLink>
            </li>
          </ul>
        </nav>
      </aside>

      <main className="mlb-content" id="center">
        <Outlet />
      </main>
    </div>
  );
};

export default Layout;
