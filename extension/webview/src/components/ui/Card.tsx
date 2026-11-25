import * as React from "react";
import { cn } from "../../lib/utils";

// Adapted from shadcndashboard/src/components/ui/card.tsx (MIT).
export function Card({ className, ...props }: React.ComponentProps<"section">) {
  return <section data-slot="card" className={cn("card", className)} {...props} />;
}

export function CardHeader({ className, ...props }: React.ComponentProps<"header">) {
  return <header data-slot="card-header" className={cn("card-header", className)} {...props} />;
}

export function CardTitle({ className, ...props }: React.ComponentProps<"h2">) {
  return <h2 data-slot="card-title" className={cn("card-title", className)} {...props} />;
}
