export type FieldSchema = { type?: string; title?: string; description?: string; default?: unknown };
export type Definition = {
  intent: string;
  display_name?: string;
  describe?: string;
  category?: string;
  domain?: string;
  aliases?: string[];
  definition_version?: string;
  assistant_status?: "active" | "paused";
  input_schema?: { properties?: Record<string, FieldSchema>; required?: string[] };
  field_policies?: Record<string, { description?: string; value_type?: string; required?: boolean; on_missing?: { action?: string; value?: unknown } }>;
  render?: { templates?: Record<string, string> };
  [key: string]: unknown;
};

export type BuilderField = {
  key: string;
  label: string;
  value_type: "string" | "number" | "integer" | "boolean";
  required: boolean;
  has_default: boolean;
  default: unknown;
};

export type BuilderSegment = { kind: "text" | "field"; value: string };
export type BuilderDraft = {
  name: string;
  description: string;
  category: string;
  examples: string[];
  fields: BuilderField[];
  segments: BuilderSegment[];
  target: "latex" | "mathjax" | "obsidian";
  intent?: string;
  base_definition?: Definition;
};

export type Activity = { id: string; action: string; intent?: string; title: string; at: string };
export type ServiceStatus = {
  state: string;
  message: string;
  python?: string;
  classifierAvailable?: boolean;
  classifierMessage?: string;
  classifierCheckpointPath?: string;
  phobertBasePath?: string;
  retrievalModelAvailable?: boolean;
  retrievalModelMessage?: string;
  retrievalModelPath?: string;
  retrievalIndexAvailable?: boolean;
  retrievalIndexMessage?: string;
  retrievalIndexPath?: string;
  modelRoot?: string;
  retrievalCacheAvailable?: boolean;
};
