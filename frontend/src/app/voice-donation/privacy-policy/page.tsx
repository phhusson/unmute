// A redirection page, set up so that we can change the URL it points to later if needed.
"use client";
import { useEffect } from "react";

const LINK =
  "https://kyutai.org/next/legal/Privacy%20Policy%20-%20Unmute%20Voice%20Donation%20Project%20v1.pdf";

export default function TermsOfUseRedirect() {
  useEffect(() => {
    window.location.href = LINK;
  }, []);
  return (
    <div>
      <p>Redirecting to Terms of Use PDF...</p>
      <a href={LINK}>Click here if not redirected.</a>
    </div>
  );
}
