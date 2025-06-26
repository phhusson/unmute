import React from "react";

const GreenLink = ({
  href,
  children,
}: {
  href: string;
  children: React.ReactNode;
}) => (
  <a
    href={href}
    target="_blank"
    rel="noopener noreferrer"
    className="text-green underline"
  >
    {children}
  </a>
);

const DonationConsent = ({
  setConsentGiven,
}: {
  setConsentGiven: (value: boolean) => void;
}) => {
  const [checks, setChecks] = React.useState([false, false, false]);

  React.useEffect(() => {
    setConsentGiven(checks.every(Boolean));
  }, [checks, setConsentGiven]);

  const handleCheck =
    (idx: number) => (e: React.ChangeEvent<HTMLInputElement>) => {
      const updated = [...checks];
      updated[idx] = e.target.checked;
      setChecks(updated);
    };

  return (
    <div className="flex flex-col gap-2 my-4">
      <label className="flex items-start gap-2">
        <input
          type="checkbox"
          checked={checks[0]}
          onChange={handleCheck(0)}
          className="mt-1.5"
        />
        <span>
          I am at least 18 years old, I am also of legal age in my country of residence and I
          have read and I agree with Kyutai’s{" "}
          <GreenLink href="/voice-donation/terms-of-use">Terms</GreenLink> and{" "}
          <GreenLink href="/voice-donation/privacy-policy">
            Privacy Policy
          </GreenLink>
          . <span className="text-red">*</span>
        </span>
      </label>
      <label className="flex items-start gap-2">
        <input
          type="checkbox"
          checked={checks[1]}
          onChange={handleCheck(1)}
          className="mt-1.5"
        />
        <span>
          I authorize Kyutai to collect, process and publish worldwide my voice
          recordings and embeddings as part of public datasets under a CC0
          license or similar open-source license, in accordance with Kyutai’s{" "}
          <GreenLink href="/voice-donation/privacy-policy">
            Privacy Policy
          </GreenLink>
          . <span className="text-red">*</span>
        </span>
      </label>
      <label className="flex items-start gap-2">
        <input
          type="checkbox"
          checked={checks[2]}
          onChange={handleCheck(2)}
          className="mt-1.5"
        />
        <span>
          I authorize Kyutai to use my voice recording and embedding worldwide
          to develop and train Kyutai’s AI models and make them available to the
          public, in accordance with Kyutai’s{" "}
          <GreenLink href="/voice-donation/privacy-policy">
            Privacy Policy
          </GreenLink>
          . <span className="text-red">*</span>
        </span>
      </label>
    </div>
  );
};

export default DonationConsent;
