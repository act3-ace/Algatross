from time import sleep

import ray


@ray.remote
class AsyncActor:
    continue_looping: bool = True
    current_thing: int = 0
    finished_first: str | None = None

    def loop_things(self) -> int:
        i = 0
        print("I'ma loopin ma things")
        while self.continue_looping:
            sleep(5)
            print(f"Loopin thing {i}")
            self.finished_first = "loop_things" if self.finished_first is None else self.finished_first
            yield i
            i += 1

    def do_something_else(self) -> int:
        self.current_thing -= 1
        self.finished_first = "do_something_else" if self.finished_first is None else self.finished_first
        return self.current_thing

    def get_finished_first(self) -> str:
        return self.finished_first

    def stop_looping(self):
        self.continue_looping = False


if __name__ == "__main__":
    remote_0 = AsyncActor.remote()
    unfinished = [remote_0.loop_things.remote()]
    remote_0.stop_looping.remote()
    while unfinished:
        print(f"something else: {ray.get(remote_0.do_something_else.remote())}")
        remote_0.stop_looping()
        futures, unfinished = ray.wait(unfinished, num_returns=1)
        print(ray.get(remote_0.get_finished_first.remote()))
        for future in futures:
            print(f"i: {ray.get(future)}")
