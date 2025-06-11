import asyncio


def get_time() -> float:
    return asyncio.get_event_loop().time()


class Stopwatch:
    def __init__(self, autostart: bool = True):
        self.start_time = get_time() if autostart else None
        self.end_time = None

    def start_if_not_started(self):
        if self.start_time is None:
            self.start_time = get_time()

    def stop(self) -> float | None:
        if self.start_time is None:
            return None

        if self.end_time is not None:
            return None  # Already stopped
        else:
            self.end_time = get_time()
            return self.end_time - self.start_time

    def time(self) -> float:
        if self.start_time is None:
            raise RuntimeError("Stopwatch not started")

        return get_time() - self.start_time

    @property
    def started(self) -> bool:
        return self.start_time is not None


class PhasesStopwatch:
    def __init__(self, phases: list[str]):
        self.phases = phases
        self.times: list[float | None] = [None for _ in phases]

    def _check_previous_phases_done(self, to: int):
        for i in range(to):
            if self.times[i] is None:
                raise RuntimeError(
                    f"Wanted to start phase {self.phases[to]} "
                    f"but earlier phase {self.phases[i]} hasn't started"
                )

    def time_phase_if_not_started(
        self, phase: str, t: float | None = None, check_previous: bool = True
    ):
        """Time a phase, either with the current time or a given time."""
        if check_previous:
            self._check_previous_phases_done(self.get_phase_index(phase))

        i = self.get_phase_index(phase)

        if self.times[i] is None:
            self.times[i] = t or get_time()

    def get_phase_index(self, phase: str) -> int:
        """Get the index of a phase."""
        try:
            i = self.phases.index(phase)
        except ValueError as e:
            raise ValueError(
                f"Phase {phase} not in phases. Valid phases: {self.phases}"
            ) from e

        return i

    def get_time_for_phase(self, phase: str) -> float:
        try:
            i = self.phases.index(phase)
        except ValueError as e:
            raise ValueError(
                f"Phase {phase} not in phases. Valid phases: {self.phases}"
            ) from e

        t = self.times[i]
        if t is None:
            raise RuntimeError(
                f"Phase {phase} not started. {self.phase_dict_partial()=}"
            )

        return t

    def phase_dict(self) -> dict[str, float]:
        return {phase: self.get_time_for_phase(phase) for phase in self.phases}

    def phase_dict_partial(self) -> dict[str, float | None]:
        return {phase: self.times[i] for i, phase in enumerate(self.phases)}

    def reset(self):
        self.times = [None for _ in self.phases]
