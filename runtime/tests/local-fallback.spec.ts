import { expect, test } from "@playwright/test";

test.describe("local fallback @webgpu", () => {
  test("demo boots with forced fallback and imports torch", async ({ page }) => {
    await page.goto("/demo/?force_fallback=1", { timeout: 180000 });

    await page.waitForFunction(
      () => Boolean((globalThis as { __torchMvpStatus?: unknown }).__torchMvpStatus),
      { timeout: 300000 },
    );

    const status = await page.evaluate(() => (globalThis as { __torchMvpStatus?: unknown }).__torchMvpStatus) as {
      ok: boolean;
      installMode: string;
      installDetail: string;
      error?: string;
    };

    expect(status.ok, status.error ?? status.installDetail).toBe(true);
    expect(status.installMode).toBe("local-dev");
  });
});
