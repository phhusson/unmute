export const getCSSVariable = (name: string) => {
  if (!name.startsWith("--")) {
    name = `--${name}`;
  }

  const variable = window
    .getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();

  if (variable === "") {
    console.warn(`CSS variable ${name} not found`);
  }
  return variable;
};
