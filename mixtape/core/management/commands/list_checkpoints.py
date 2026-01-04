from django.db.models import Count
from django.utils import timezone
import djclick as click

from mixtape.core.models.checkpoint import Checkpoint


@click.command()
def list_checkpoints() -> None:
    """List available checkpoints."""
    checkpoints = Checkpoint.objects.select_related('training')

    checkpoints = checkpoints.annotate(
        inference_count=Count('inference', distinct=True),
        episode_count=Count('inference__episode', distinct=True),
    ).order_by('-created')

    checkpoints_list = list(checkpoints)

    headers = ('environment', 'checkpoint_pk', 'created', 'inferences', 'episodes')
    rows: list[tuple[str, str, str, str, str]] = []

    for checkpoint in checkpoints_list:
        created_local = timezone.localtime(checkpoint.created)
        rows.append(
            (
                checkpoint.training.environment,
                str(checkpoint.pk),
                created_local.strftime('%Y-%m-%d %H:%M:%S'),
                str(checkpoint.inference_count),
                str(checkpoint.episode_count),
            )
        )

    if not rows:
        click.echo('No checkpoints found.')
        return

    col_widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(value))

    def format_row(values: tuple[str, ...]) -> str:
        return ' | '.join(value.ljust(col_widths[idx]) for idx, value in enumerate(values))

    header_line = format_row(headers)
    separator_line = '-+-'.join('-' * width for width in col_widths)

    click.echo(header_line)
    click.echo(separator_line)
    for row in rows:
        click.echo(format_row(row))
