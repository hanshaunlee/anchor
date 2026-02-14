import { SessionDetailContent } from "./session-detail-content";

/** Resolve params (Next 14: object, Next 15: Promise) so we never use React.use(Promise) in a client component. */
async function resolveParams(params: Promise<{ id: string }> | { id: string }): Promise<{ id: string }> {
  if (typeof (params as Promise<unknown>)?.then === "function") {
    return params as Promise<{ id: string }>;
  }
  return params as { id: string };
}

export default async function SessionDetailPage({
  params,
}: {
  params: Promise<{ id: string }> | { id: string };
}) {
  const { id } = await resolveParams(params);
  return <SessionDetailContent id={id} />;
}
