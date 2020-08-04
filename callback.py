class myCallback(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):
    sayi = self.params["epochs"]
    if(epoch % 100 == 0):

      sys.stdout.write(f"\rLoss: {logs.get('loss')} | Epoch: {epoch} | Completed: % { int((epoch/sayi)*100) } ")
      sys.stdout.flush()
      
    if(logs.get("loss") < 0.10):
      print("Model has < 0.10 Loss, training stopped.")
      self.model.stop_training = True