import { Outlet, NavLink, useLocation } from "react-router-dom";
import {
  LayoutDashboard,
  LineChart,
  GitCompare,
  List,
  Star,
  BadgeCheck,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

/* ── Navigation data ──────────────────────────────────────────────── */

interface NavItem {
  name: string;
  path: string;
  icon: LucideIcon;
}

interface NavGroup {
  section: string;
  links: NavItem[];
}

const sections = [
  { name: "Research", path: "/research", defaultPath: "/research/latest" },
  { name: "History", path: "/history", defaultPath: "/history/ledger" },
  { name: "Promotion", path: "/promotion", defaultPath: "/promotion" },
] as const;

const navGroups: NavGroup[] = [
  {
    section: "Research",
    links: [
      { name: "Latest Run", path: "/research/latest", icon: LayoutDashboard },
      { name: "Stage Cards", path: "/research/stages", icon: LineChart },
      { name: "Benchmark", path: "/research/benchmark", icon: GitCompare },
    ],
  },
  {
    section: "History",
    links: [
      { name: "Run Ledger", path: "/history/ledger", icon: List },
      { name: "Best Runs", path: "/history/best", icon: Star },
    ],
  },
  {
    section: "Promotion",
    links: [
      { name: "Summary", path: "/promotion", icon: BadgeCheck },
    ],
  },
];

const mobileNavItems: (NavItem & { matchPrefix: string })[] = [
  { name: "Research", path: "/research/latest", matchPrefix: "/research", icon: LayoutDashboard },
  { name: "History", path: "/history/ledger", matchPrefix: "/history", icon: List },
  { name: "Promotion", path: "/promotion", matchPrefix: "/promotion", icon: BadgeCheck },
];

/* ── Layout shell ─────────────────────────────────────────────────── */

export default function Layout() {
  const location = useLocation();

  return (
    <div className="min-h-screen bg-slab text-ink font-sans">
      {/* ── Top Bar ─────────────────────────────────────────────── */}
      <header className="fixed top-0 w-full z-50 bg-slab flex items-center justify-between px-6 h-16 border-b border-stroke/15">
        <div className="flex items-center gap-4">
          <span className="text-xl font-extrabold text-accent tracking-tighter font-heading">
            The Precision Analyst
          </span>

          <nav className="hidden md:flex items-center ml-8 gap-6 h-full">
            {sections.map((section) => {
              const active = location.pathname.startsWith(section.path);
              return (
                <NavLink
                  key={section.path}
                  to={section.defaultPath}
                  className={`h-16 flex items-center font-heading font-bold tracking-tight transition-colors duration-200 ${
                    active
                      ? "text-accent border-b-2 border-accent"
                      : "text-ink-dim hover:text-accent"
                  }`}
                >
                  {section.name}
                </NavLink>
              );
            })}
          </nav>
        </div>
      </header>

      {/* ── Side Nav (desktop) ──────────────────────────────────── */}
      <aside className="fixed left-0 top-16 h-[calc(100vh-64px)] w-64 border-r border-stroke/15 bg-slab hidden md:flex flex-col py-6">
        {/* User info card */}
        <div className="px-4 mb-8">
          <div className="flex items-center gap-3 p-3 glass-panel rounded-xl">
            <div className="w-10 h-10 rounded-full bg-deep flex items-center justify-center text-accent font-bold">
              PA
            </div>
            <div>
              <p className="text-ink font-bold text-sm">Analyst Pro</p>
              <p className="text-ink-dim text-[10px] font-bold uppercase tracking-widest">
                Personal Edition
              </p>
            </div>
          </div>
        </div>

        {/* Grouped nav links */}
        <nav className="flex-1 flex flex-col gap-6 overflow-y-auto">
          {navGroups.map((group) => (
            <div key={group.section}>
              <p className="px-4 mb-2 text-ink-dim text-[10px] font-bold uppercase tracking-widest">
                {group.section}
              </p>
              {group.links.map((link) => (
                <NavLink
                  key={link.path}
                  to={link.path}
                  className={({ isActive }) =>
                    `flex items-center gap-3 px-4 py-3 text-sm font-medium uppercase tracking-wider transition-all duration-300 hover:translate-x-1 ${
                      isActive
                        ? "text-accent bg-panel rounded-r-lg border-l-4 border-accent"
                        : "text-ink-dim hover:bg-panel/50"
                    }`
                  }
                >
                  <link.icon size={18} />
                  <span>{link.name}</span>
                </NavLink>
              ))}
            </div>
          ))}
        </nav>
      </aside>

      {/* ── Main content ────────────────────────────────────────── */}
      <main className="pt-16 pb-20 md:pb-0 md:pl-64">
        <div className="p-6">
          <Outlet />
        </div>
      </main>

      {/* ── Mobile bottom nav ───────────────────────────────────── */}
      <nav className="md:hidden fixed bottom-0 w-full h-16 bg-slab flex justify-around items-center z-50 border-t border-stroke/15">
        {mobileNavItems.map((item) => {
          const active = location.pathname.startsWith(item.matchPrefix);
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={`flex flex-col items-center gap-1 ${
                active ? "text-accent" : "text-ink-dim"
              }`}
            >
              <item.icon size={20} />
              <span className="text-[10px] font-bold uppercase">{item.name}</span>
            </NavLink>
          );
        })}
      </nav>
    </div>
  );
}
