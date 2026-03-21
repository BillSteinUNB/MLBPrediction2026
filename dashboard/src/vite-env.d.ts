/// <reference types="vite/client" />

/**
 * plotly.js-basic-dist ships without its own type declarations.
 * Re-export the full plotly.js types so consumers get autocomplete
 * and type-safety while still pulling the smaller bundle at runtime.
 */
declare module "plotly.js-basic-dist" {
  import Plotly from "plotly.js";
  export = Plotly;
}
