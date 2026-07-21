class RegressionDetector:

    @staticmethod
    def compare(previous_run, current_run):

        previous_accuracy = previous_run[4]
        current_accuracy = current_run[4]

        difference = current_accuracy - previous_accuracy

        if difference > 0:
            status = "improved"

        elif difference < 0:
            status = "regression"

        else:
            status = "unchanged"

        return {
            "status": status,
            "difference": difference,
            "previous_accuracy": previous_accuracy,
            "current_accuracy": current_accuracy,
            "previous_run": previous_run[0],
            "current_run": current_run[0],
            "previous_timestamp": previous_run[1],
            "current_timestamp": current_run[1],
            "prompt_version": current_run[2],
            "model": current_run[3],
        }