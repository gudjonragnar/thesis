import torch
import params

from ignite_softmax_sccnn import softmaxSCCNN
from data import NEPValidationDataset, ClassificationDataset
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy
from torch.utils.data import DataLoader

def handler(output):
    pred, target = output
    pred = torch.sum(pred, dim=0)
    stack = [pred]*len(target)
    return torch.stack(stack), target

model = softmaxSCCNN()
model.save_model()
nep_evaluator = create_supervised_evaluator(model, metrics={'acc': Accuracy(handler)})
evaluator = create_supervised_evaluator(model, metrics={'acc': Accuracy()})

nep_test_ds = NEPValidationDataset(root_dir=params.root_dir, d=3)
nep_test_dl = DataLoader(nep_test_ds, batch_size=len(nep_test_ds.shifts), num_workers=params.num_workers, drop_last=True)

test_ds = ClassificationDataset(root_dir=params.root_dir, train=False)
test_dl = DataLoader(test_ds, batch_size=params.batch_size, num_workers=params.num_workers)

# it = iter(test_dl)
# print(next(it)[1])

test_output = torch.tensor([
    [.1, .1, .5, .3],
    [.1, .1, .5, .3],
    [.1, .1, .5, .3]
])
test_target = torch.tensor([1,1,1])

# print(handler((test_output,test_target)))
evaluator.run(test_dl)
nep_evaluator.run(nep_test_dl)
metrics = evaluator.state.metrics
nep_metrics = nep_evaluator.state.metrics
print('Single patch accuracy: {}, NEP accuracy: {}'.format(metrics['acc'], nep_metrics['acc']))

print('Loading trained model!')
# model.load_model('checkpoints/sccnn_model_120.pth')
model.load_model('model/statedict.pth')
print('Loading finished!')

evaluator.run(test_dl)
nep_evaluator.run(nep_test_dl)
metrics = evaluator.state.metrics
nep_metrics = nep_evaluator.state.metrics
print('Single patch accuracy: {}, NEP accuracy: {}'.format(metrics['acc'], nep_metrics['acc']))