"use client";

import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FileText, TrendingDown, ShieldAlert } from "lucide-react";

export default function ReportsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Reports</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Agent-generated reports. Run the corresponding agent from the Agents page to produce data, then open the report below or use the &quot;View report&quot; button on the agent card.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <Card className="rounded-2xl shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <FileText className="h-4 w-4" />
              Evidence Narrative
            </CardTitle>
            <p className="text-muted-foreground text-sm">
              Caregiver narrative, elder-safe view, and hypotheses per signal. Run the Evidence Narrative agent, then use &quot;View report&quot; on the Agents page to open the latest report.
            </p>
          </CardHeader>
          <CardContent>
            <Link href="/agents">
              <Button variant="outline" size="sm">Go to Agents</Button>
            </Link>
          </CardContent>
        </Card>

        <Card className="rounded-2xl shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <TrendingDown className="h-4 w-4" />
              Calibration report
            </CardTitle>
            <p className="text-muted-foreground text-sm">
              Before/after ECE, precision and recall from the Continual Calibration agent. Opens the latest run&apos;s report.
            </p>
          </CardHeader>
          <CardContent>
            <Link href="/reports/calibration">
              <Button variant="outline" size="sm">View calibration report</Button>
            </Link>
          </CardContent>
        </Card>

        <Card className="rounded-2xl shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <ShieldAlert className="h-4 w-4" />
              Red-team report
            </CardTitle>
            <p className="text-muted-foreground text-sm">
              Regression pass rate, failing cases, and &quot;Open in replay&quot; from the Synthetic Red-Team agent. Opens the latest run&apos;s report.
            </p>
          </CardHeader>
          <CardContent>
            <Link href="/reports/redteam">
              <Button variant="outline" size="sm">View red-team report</Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
