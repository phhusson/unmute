import type { MDXComponents } from "mdx/types";

// This file allows you to provide custom React components
// to be used in MDX files. You can import and use any
// React component you want, including inline styles,
// components from other libraries, and more.

export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    h1: ({ children }) => (
      <h1 className="text-3xl mt-8 text-white">{children}</h1>
    ),
    h2: ({ children }) => (
      <h1 className="text-2xl font-semibold mt-6 text-white">{children}</h1>
    ),
    h3: ({ children }) => (
      <h1 className="text-xl font-semibold mt-4 text-white">{children}</h1>
    ),
    p: ({ children }) => (
      <p className="text-sm md:text-base my-2">{children}</p>
    ),
    li: ({ children }) => (
      <li className="text-sm md:text-base my-2 ml-6 list-disc">{children}</li>
    ),
    a: ({ href, children }) => (
      <a className="text-green underline" href={href}>
        {children}
      </a>
    ),
    strong: ({ children }) => (
      <strong className="text-white">{children}</strong>
    ),
    // TODO more styling here
    ...components,
  };
}
