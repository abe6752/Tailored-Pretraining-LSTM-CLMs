import sys
from typing import Optional
from ..generative_models.training import GenerativeModelTrainer

class TrainerApp():
	def __init__(self, argv: Optional[str]=None):
		if argv is None:
			argv = sys.argv[1:]
		self.argv = argv
	def main(self):
		trainer = GenerativeModelTrainer(self.argv)
		trainer.main()
		
if __name__ == '__main__':
	TrainerApp().main()
			