import * as React from "react";
import { cn } from "../../lib/utils";

// Table primitives use the same data-slot composition as the MIT dashboard source.
export function Table({ className, ...props }: React.ComponentProps<"table">) {
  return <div data-slot="table-container" className="table-wrap"><table data-slot="table" className={cn("data-table", className)} {...props} /></div>;
}

export function TableHead({ className, ...props }: React.ComponentProps<"th">) {
  return <th data-slot="table-head" className={cn("table-head", className)} {...props} />;
}

export function TableCell({ className, ...props }: React.ComponentProps<"td">) {
  return <td data-slot="table-cell" className={cn("table-cell", className)} {...props} />;
}
