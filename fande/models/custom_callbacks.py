from pytorch_lightning import Callback


class MyCallbacks(Callback):

      def setup(self, trainer, pl_module, stage=None):
            print("\n setup() callback called...")

            # if pl_module.hparams["device"] == "gpu":
            #       pl_module = pl_module.cuda()

      def teardown(self, trainer, pl_module, stage=None):
            print("\n teardown() callback called...")

            if pl_module.hparams["device"] == "gpu":
                  pl_module = pl_module.cuda()

      def on_predict_start(self, trainer, pl_module):
            print("\n on_predict_start() callback called...")

            if pl_module.hparams["device"] == "gpu":
                  pl_module = pl_module.cuda()

      # def on_train_end(self, trainer, pl_module):
      #     print("do something when training ends")
      #     pl_module = pl_module.cuda()

      # def on_fit_end(self, trainer, pl_module):

      #       print("\n on_fit_end command called")

      #       if pl_module.hparams["device"] == "gpu":
      #             pl_module = pl_module.cuda()
