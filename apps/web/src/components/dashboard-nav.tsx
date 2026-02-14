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
  Upload,
} from "lucide-react";

const nav = [
  { href: "/dashboard", label: "Today", icon: LayoutDashboard },
  { href: "/alerts", label: "Alerts", icon: AlertTriangle },
  { href: "/graph", label: "Graph view", icon: Network },
  { href: "/sessions", label: "Sessions", icon: Calendar },
  { href: "/watchlists", label: "Watchlists", icon: List },
  { href: "/summaries", label: "Summaries", icon: FileText },
  { href: "/agents", label: "Agents", icon: Bot },
  { href: "/ingest", label: "Ingest events", icon: Upload },
  { href: "/replay", label: "Scenario Replay", icon: Play },
];
const elderNav = { href: "/elder", label: "Elder view", icon: User };

export function DashboardNav() {
  const pathname = usePathname();
  return (
    <nav className="flex flex-col gap-1">
      {nav.map((item) => {
        const Icon = item.icon;
        const active = pathname === item.href || (item.href !== "/dashboard" && pathname.startsWith(item.href));
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
      <div className="my-2 border-t border-border" />
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
    </nav>
  );
}
