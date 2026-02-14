import Link from "next/link";

export default function HomePage() {
  // In a real app we'd check auth server-side; for now we redirect to dashboard
  // and let middleware or client handle login redirect
  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-8 p-8 bg-[#F8F8F5]">
      <img src="/logo.png" alt="Anchor" className="h-24 w-auto sm:h-28" />
      <h1 className="sr-only">Anchor</h1>
      <p className="text-muted-foreground text-center max-w-md">
        Elder companion & graph risk engine. Offline-first, privacy-aware.
      </p>
      <div className="flex flex-wrap gap-4 justify-center">
        <Link
          href="/signup"
          className="rounded-2xl bg-primary px-6 py-3 text-sm font-medium text-primary-foreground shadow-sm transition hover:bg-primary/90"
        >
          Create account
        </Link>
        <Link
          href="/login"
          className="rounded-2xl border border-border bg-background px-6 py-3 text-sm font-medium shadow-sm transition hover:bg-accent"
        >
          Sign in
        </Link>
        <Link
          href="/dashboard"
          className="rounded-2xl border border-border bg-background px-6 py-3 text-sm font-medium shadow-sm transition hover:bg-accent"
        >
          Dashboard
        </Link>
      </div>
      <Link href="/replay" className="text-sm text-muted-foreground hover:underline">
        Scenario Replay (demo)
      </Link>
    </div>
  );
}
