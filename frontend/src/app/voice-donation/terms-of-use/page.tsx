// A redirection page, set up so that we can change the URL it points to later if needed.
"use client";
import { useEffect } from "react";

const LINK =
  "https://kyutai.org/next/legal/Terms%20of%20Use%20-%20Unmute%20Voice%20Donation%20Project%20v1.pdf";

export default function PrivacyPolicyRedirect() {
  useEffect(() => {
    window.location.href = LINK;
  }, []);
  return (
    <div>
      <p>Redirecting to Privacy Policy PDF...</p>
      <a href={LINK}>Click here if not redirected.</a>
    </div>
  );
}
