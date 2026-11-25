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
  target: string;
  intent?: string;
  base_definition?: Definition;
};

export type CategoryOption = { value: string; label: string };
export type MilestoneEvalMetrics = {
  val_top1?: number;
  val_recall3?: number;
  total?: number;
};
export type MilestoneTrainingReport = {
  timestamp?: string;
  device?: string;
  epochs?: number;
  elapsed_seconds?: number;
  deployed?: boolean;
  before_val?: MilestoneEvalMetrics;
  after_val?: MilestoneEvalMetrics;
  dataset?: { train_rows?: number; val_rows?: number; selected_intents?: number; user_intents?: number };
};
export type MilestoneStatus = {
  pending_intents?: string[];
  pending_count?: number;
  threshold?: number;
  milestone_ready?: boolean;
  training_running?: boolean;
  last_error?: string;
  last_trained_at?: string | null;
  feedback_count?: number;
  last_report?: MilestoneTrainingReport | null;
};

export type Activity = { id: string; action: string; intent?: string; title: string; at: string };
export type ServiceStatus = {
  state: string;
  message: string;
  python?: string;
  pipelineMode?: string;
  classifierAvailable?: boolean;
  classifierMissingDependencies?: string[];
  classifierMessage?: string;
  classifierCheckpointPath?: string;
  phobertBasePath?: string;
  retrievalModelAvailable?: boolean;
  retrievalMissingDependencies?: string[];
  retrievalModelMessage?: string;
  retrievalModelPath?: string;
  retrievalIndexAvailable?: boolean;
  retrievalIndexMessage?: string;
  retrievalIndexPath?: string;
  modelRoot?: string;
  retrievalCacheAvailable?: boolean;
};
