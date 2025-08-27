from datetime import datetime
from pync import Notifier

class Timer:
    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, *args):
        self.end_time = datetime.now()
        self.interval = (self.end_time - self.start_time).total_seconds()
        self.interval = round(self.interval, 2)

        now_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Show in hours, minutes and seconds if it took more than a minute
        if self.interval > 3600:
            hours = int(self.interval / 3600)
            minutes = int((self.interval % 3600) / 60)
            seconds = self.interval % 60
            print(
                f"[{now_timestamp}] Ran in {hours} hours, {minutes} minutes and {seconds} seconds."
            )
            Notifier.notify(
                f"Ran in {hours} hours, {minutes} minutes and {seconds} seconds.",
            )

        elif self.interval > 60:
            minutes = int(self.interval / 60)
            seconds = int(self.interval % 60)
            print(f"[{now_timestamp}] Ran in {minutes} minutes and {seconds} seconds.")
            Notifier.notify(
                f"Ran in {minutes} minutes and {seconds} seconds.",
            )

        else:
            print(f"[{now_timestamp}] Ran in {self.interval} seconds.")
            Notifier.notify(f"Ran in {self.interval} seconds.")
