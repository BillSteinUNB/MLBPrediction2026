import React from "react";
import { NavLink, Outlet } from "react-router-dom";
import "./layout.css";

const Layout: React.FC = () => {
  return (
    <div className="mlb-layout">
      <aside className="mlb-sidebar" role="navigation" aria-label="Main">
        <div className="mlb-brand">MLB Live Dashboard</div>
        <nav>
          <ul className="mlb-nav">
            <li>
              <NavLink to="/" end className={({ isActive }) => (isActive ? "active" : "")}>
                Overview
              </NavLink>
            </li>
            <li>
              <NavLink to="/slate" className={({ isActive }) => (isActive ? "active" : "")}>
                Live Slate
              </NavLink>
            </li>
            <li>
              <NavLink to="/live-season" className={({ isActive }) => (isActive ? "active" : "")}>
                Live Season
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
