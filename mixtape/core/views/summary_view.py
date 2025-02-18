from django.shortcuts import render

from mixtape.core.models.step import Step


def simple_view(request, step_id):
    step = Step.objects.get(step_id)
    return render(request, "simple_view.html", {"step": step})
