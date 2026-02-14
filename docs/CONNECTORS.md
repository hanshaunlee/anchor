# Bank Connectors & Deployment Realism

## Overview

Anchor's **Incident Response Agent** is capability-aware: it only executes actions that are explicitly supported by the household's connected providers. We **never overpromise** lock/freeze or account control unless the connector returned success and we recorded a receipt.

## What Plaid / Open Banking Enables

- **Plaid** (and similar open-banking APIs) typically provide:
  - **Read access**: account list, balances, transaction history (with user consent and linking flow).
  - **No universal control**: most banks do not expose “lock card” or “disable transfers” via Plaid. A few institutions offer limited controls (e.g. enable alerts) through provider-specific APIs.

- **Open Banking** (PSD2/UK Open Banking, etc.):
  - Primarily **read-only** (accounts, transactions).
  - **Optional control** is institution- and region-specific; not guaranteed.

## What Is Optional Control and Why It’s Rare

- **lock_card**, **disable_transfers**, **open_dispute**: These require direct bank APIs or partner agreements. Most integrations are read-only.
- **enable_alerts**: Sometimes available via Plaid or bank-specific endpoints; we gate it behind `household_capabilities.bank_control_capabilities.enable_alerts`.
- We expose a **Capability Registry** (`household_capabilities`): each household has explicit flags. The UI and agent only show or execute actions when the corresponding capability is `true`. Otherwise we produce **guided playbooks** (call script, email template, case file) for the caregiver to execute manually.

## How We Avoid Overpromising

1. **Explicit capabilities**: `household_capabilities.bank_control_capabilities` lists what the household’s bank integration supports. Defaults are conservative (e.g. `lock_card: false`).
2. **Incident Response Agent**:
   - Builds an **Action DAG** from capabilities: e.g. `LOCK_CARD` task only if `lock_card` is true; otherwise `FREEZE_CARD_INSTRUCTION` with a call script.
   - Never claims to have locked/frozen anything unless the connector returned success and we stored a receipt.
3. **UI**: Buttons that depend on a capability show “Not available” with a short reason (e.g. “Your bank integration does not support lock card; we prepared the call script instead.”) when the capability is false.

## Security Notes

- **Tokens**: Never store raw bank credentials. Store access tokens encrypted (or in Supabase Vault if available). For hackathon/demo, tokens may live in env or service role only; document this and move to secure storage for production.
- **Consent**: Outbound contact (notify caregiver, call/email) is gated by `allow_outbound_contact` and role (caregiver/admin for triggering). Elder view is simplified and consent-gated.
- **Logs**: Do not log full account numbers or tokens. Log household_id, connector type, and success/failure only where needed for support.

## Connector Implementation Status

| Connector        | Data (read)        | Controls              | Notes                          |
|-----------------|--------------------|------------------------|--------------------------------|
| **Mock**        | Events/entities or fixtures | None                 | Demo/hackathon; no real bank   |
| **Plaid**       | Stub (501 if no creds) | Stub                | Implement link_token, exchange, sync when creds set |
| **Open Banking**| Not implemented   | N/A                    | Future                         |

## API Endpoints

- **GET /capabilities/me** – Current household capabilities.
- **PATCH /capabilities** – Update capabilities (caregiver/admin); demo config.
- **GET /connectors/plaid/link_token** – Returns 501 with reason if Plaid not configured.
- **POST /connectors/plaid/exchange_public_token** – 501 if not configured.
- **POST /connectors/plaid/sync_transactions** – 501 if not configured.

When a connector is not configured, the Incident Response Agent falls back to **event-derived financial context** (e.g. from `transaction_detected`, `payee_added` events) and still produces the playbook and bank-ready case file.
