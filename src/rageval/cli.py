"""Command line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from rageval.config import PipelineConfig, build_chunker_from
from rageval.corpus import build_corpus, load_questions, validate_dataset
from rageval.harness import gate as gate_module
from rageval.harness import report as report_module
from rageval.harness.runner import Pipeline
from rageval.metrics.judge import calibrate, load_human_labels

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Evaluate, compare and gate retrieval-augmented generation pipelines.",
)
console = Console()

ConfigArg = Annotated[Path, typer.Argument(help="Path to a pipeline YAML config.")]


@app.command()
def run(
    config: ConfigArg,
    out: Annotated[Path, typer.Option(help="Where to write the run JSON.")] = Path("runs"),
    run_id: Annotated[str | None, typer.Option(help="Identifier for this run.")] = None,
    full: Annotated[bool, typer.Option(help="Report every metric, not just headlines.")] = False,
) -> None:
    """Run one pipeline over the golden set and report its metrics."""
    cfg = PipelineConfig.load(config)
    pipeline = Pipeline.build(cfg)
    result = pipeline.run(run_id=run_id)

    destination = out / f"{result.run_id}.json" if out.suffix == "" else out
    report_module.save_run(result, destination)
    console.print(report_module.render_run(result, headline_only=not full))
    console.print(f"[dim]wrote {destination}[/dim]")


@app.command()
def compare(
    baseline: Annotated[Path, typer.Argument(help="Baseline run JSON.")],
    candidate: Annotated[Path, typer.Argument(help="Candidate run JSON.")],
    metric: Annotated[list[str] | None, typer.Option(help="Restrict to these metrics.")] = None,
) -> None:
    """Compare two runs with a paired significance test."""
    base = report_module.load_run(baseline)
    cand = report_module.load_run(candidate)
    comparisons = report_module.compare_runs(base, cand, metrics=metric or None)
    console.print(report_module.render_comparison(base, cand, comparisons))


@app.command()
def gate(
    candidate: Annotated[Path, typer.Argument(help="Candidate run JSON.")],
    gates: Annotated[Path, typer.Option(help="Gate threshold YAML.")] = Path("configs/gates.yaml"),
    baseline: Annotated[Path | None, typer.Option(help="Baseline run JSON.")] = None,
) -> None:
    """Fail the build when a run breaches a floor or regresses against the baseline."""
    cfg = gate_module.GateConfig.load(gates)
    cand = report_module.load_run(candidate)
    base = report_module.load_run(baseline) if baseline and baseline.exists() else None
    if baseline and base is None:
        console.print(f"[yellow]no baseline at {baseline}; checking absolute floors only[/yellow]")

    violations = gate_module.evaluate(cand, base, cfg)
    if not violations:
        console.print("[green]gate passed[/green]")
        return
    console.print(f"[red]gate failed with {len(violations)} violation(s)[/red]")
    for violation in violations:
        console.print(f"  [red]x[/red] {violation.metric}: {violation.reason}")
    raise typer.Exit(code=1)


@app.command()
def ablate(
    configs: Annotated[list[Path], typer.Argument(help="Pipeline configs to sweep.")],
    out: Annotated[Path, typer.Option(help="Directory for run JSON files.")] = Path("runs"),
    against: Annotated[
        str | None, typer.Option(help="Pipeline name to use as the comparison baseline.")
    ] = None,
    report: Annotated[Path | None, typer.Option(help="Write the markdown report here.")] = None,
) -> None:
    """Run several pipelines and rank them, with paired tests against a baseline."""
    results = []
    for path in configs:
        cfg = PipelineConfig.load(path)
        console.print(f"[dim]running {cfg.name}[/dim]")
        result = Pipeline.build(cfg).run()
        report_module.save_run(result, out / f"{result.run_id}.json")
        results.append(result)

    sections = ["# Ablation", "", report_module.render_leaderboard(results)]
    baseline_name = against or results[0].pipeline
    base = next((r for r in results if r.pipeline == baseline_name), None)
    if base is not None:
        for result in results:
            if result.pipeline == base.pipeline:
                continue
            comparisons = report_module.compare_runs(
                base, result, metrics=list(report_module.HEADLINE)
            )
            sections += ["", report_module.render_comparison(base, result, comparisons)]

    text = "\n".join(sections)
    console.print(text)
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(text, encoding="utf-8")
        console.print(f"[dim]wrote {report}[/dim]")


@app.command("validate")
def validate(config: ConfigArg) -> None:
    """Check that every gold quote resolves and is reachable under this chunking."""
    cfg = PipelineConfig.load(config)
    corpus = build_corpus(cfg.dataset.corpus_dir, build_chunker_from(cfg.chunker))
    questions = load_questions(cfg.dataset.questions)
    problems = validate_dataset(corpus, questions)

    console.print(
        f"{len(corpus.documents)} documents, {len(corpus.chunks)} chunks, "
        f"{len(questions)} questions"
    )
    if not problems:
        console.print("[green]dataset is sound[/green]")
        return
    console.print(f"[red]{len(problems)} problem(s)[/red]")
    for problem in problems:
        console.print(f"  [red]x[/red] {problem}")
    raise typer.Exit(code=1)


@app.command("calibrate-judge")
def calibrate_judge(
    config: ConfigArg,
    labels: Annotated[
        Path | None, typer.Option(help="Human label JSONL; defaults to the dataset's.")
    ] = None,
) -> None:
    """Measure how far the configured judge agrees with human labels."""
    cfg = PipelineConfig.load(config)
    label_path = labels or cfg.dataset.human_labels
    if label_path is None:
        raise typer.BadParameter("no human labels configured for this dataset")

    human = load_human_labels(label_path)
    pipeline = Pipeline.build(cfg)
    result = pipeline.run()

    paired = [
        (int(scores["judge_correct"]), human[qid])
        for qid, scores in sorted(result.per_question.items())
        if qid in human and "judge_correct" in scores
    ]
    if not paired:
        raise typer.BadParameter("no overlap between human labels and scored questions")

    report = calibrate([j for j, _ in paired], [h for _, h in paired])
    console.print(f"[bold]judge:[/bold] {cfg.judge.name}  [bold]n:[/bold] {report.n}")
    console.print(f"agreement      {report.agreement:.3f}")
    console.print(f"cohen's kappa  {report.kappa:.3f}")
    console.print(
        f"judge says correct {report.judge_positive_rate:.3f}, "
        f"humans say correct {report.human_positive_rate:.3f}"
    )
    console.print(
        f"false positives {report.false_positives}, false negatives {report.false_negatives}"
    )
    console.print(f"\n[bold]{report.verdict}[/bold]")


if __name__ == "__main__":
    app()
