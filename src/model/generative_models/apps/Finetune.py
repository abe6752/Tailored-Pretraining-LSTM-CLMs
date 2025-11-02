import sys
from typing import Optional
from ..generative_models.finetune import Finetuner

class FinetunerApp():
	def __init__(self, argv: Optional[str]=None):
		if argv is None:
			argv = sys.argv[1:]
		self.argv = argv

	def main(self):
		tuner = Finetuner(self.argv)
		tuner.main()

if __name__ == '__main__':
	FinetunerApp().main()
