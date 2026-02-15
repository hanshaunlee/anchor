"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  LayoutDashboard,
  AlertTriangle,
  Calendar,
  List,
  FileText,
  Bot,
  User,
  Play,
  Network,
  CircleDot,
  ClipboardList,
  Upload,
  Settings,
  Shield,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import { useState } from "react";

const overviewNav = [
  { href: "/dashboard", label: "Today", icon: LayoutDashboard },
  { href: "/alerts", label: "Alerts", icon: AlertTriangle },
];

const protectionNav = [
  { href: "/protection", label: "Protection", icon: Shield },
  { href: "/graph", label: "Graph view", icon: Network },
  { href: "/sessions", label: "Sessions", icon: Calendar },
];

const automationNav = [
  { href: "/agents", label: "Automation Center", icon: Bot },
  { href: "/replay", label: "Scenario Replay", icon: Play },
];

const advancedNav = [
  { href: "/watchlists", label: "Watchlists", icon: List },
  { href: "/rings", label: "Rings", icon: CircleDot },
  { href: "/reports", label: "Reports", icon: ClipboardList },
];

const otherNav = [
  { href: "/summaries", label: "Summaries", icon: FileText },
  { href: "/ingest", label: "Ingest events", icon: Upload },
];

const elderNav = { href: "/elder", label: "Elder view", icon: User };

function NavGroup({
  title,
  items,
  pathname,
  defaultOpen = true,
}: {
  title: string;
  items: { href: string; label: string; icon: React.ComponentType<{ className?: string }> }[];
  pathname: string;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const hasActive = items.some(
    (item) => pathname === item.href || (item.href !== "/dashboard" && pathname.startsWith(item.href))
  );
  return (
    <div className="space-y-1">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between rounded-xl px-3 py-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground hover:text-foreground"
      >
        {title}
        {open ? <ChevronDown className="h-3.5 w-3" /> : <ChevronRight className="h-3.5 w-3" />}
      </button>
      {open && (
        <div className="space-y-0.5">
          {items.map((item) => {
            const Icon = item.icon;
            const active =
              pathname === item.href || (item.href !== "/dashboard" && pathname.startsWith(item.href));
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition",
                  active ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-accent hover:text-foreground"
                )}
              >
                <Icon className="h-4 w-4 shrink-0" />
                {item.label}
              </Link>
            );
          })}
        </div>
      )}
    </div>
  );
}

export function DashboardNav() {
  const pathname = usePathname();
  return (
    <nav className="flex flex-col gap-4">
      <NavGroup title="Overview" items={overviewNav} pathname={pathname} />
      <NavGroup title="Protection" items={protectionNav} pathname={pathname} />
      <NavGroup title="Automation" items={automationNav} pathname={pathname} />
      <NavGroup title="Advanced" items={advancedNav} pathname={pathname} defaultOpen={false} />
      <div className="space-y-0.5">
        {otherNav.map((item) => {
          const Icon = item.icon;
          const active =
            pathname === item.href || (item.href !== "/dashboard" && pathname.startsWith(item.href));
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition",
                active ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-accent hover:text-foreground"
              )}
            >
              <Icon className="h-4 w-4 shrink-0" />
              {item.label}
            </Link>
          );
        })}
      </div>
      <div className="border-t border-border pt-2 mt-2">
        <Link
          href={elderNav.href}
          className={cn(
            "flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition",
            pathname === elderNav.href
              ? "bg-primary text-primary-foreground"
              : "text-muted-foreground hover:bg-accent hover:text-foreground"
          )}
        >
          <elderNav.icon className="h-4 w-4 shrink-0" />
          {elderNav.label}
        </Link>
        <Link
          href="/settings"
          className={cn(
            "flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition",
            pathname === "/settings"
              ? "bg-primary text-primary-foreground"
              : "text-muted-foreground hover:bg-accent hover:text-foreground"
          )}
        >
          <Settings className="h-4 w-4 shrink-0" />
          Settings
        </Link>
      </div>
    </nav>
  );
}
