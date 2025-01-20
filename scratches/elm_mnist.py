import time

import numpy as np

import torch

# from torchvision.models import ResNet34_Weights, resnet34
import scipy
import torch_tensorrt  # noqa: F401

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from algatross.extreme_learning_machines.modules.linear import ELMHiddenLinear, ELMLinear
from algatross.extreme_learning_machines.modules.sequential import ELMSimpleSequential
from algatross.extreme_learning_machines.utils.ops import imqrginv
from algatross.models.elm import MLGELMEncoder

# train = pd.read_csv("datasets/mnist_train.csv")
# test = pd.read_csv("datasets/mnist_test.csv")


device = "cuda"
# device = "cpu"
torch_dtype = torch.float64
# torch_dtype = torch.float128
numpy_dtype = np.float64


varest_method = "interquantile_range"
varest_config = {"range_begin": 1.0 / 14.0, "range_end": 13.0 / 14.0}
# varest_config = {"range_begin": 0.25, "range_end": 0.75}
weighting_interpolation_method = "linear"
weighting_interpolation_config = {"c_1": 2.5, "c_2": 3.0}

chunk_size = 10000

gamma = 100

elm_learning_config = {
    "varest_method": varest_method,
    "varest_config": varest_config,
    "weighting_interpolation_config": weighting_interpolation_config,
    "weighting_interpolation_method": weighting_interpolation_method,
    "risk_gamma": gamma,
    "weighted": False,
}

X_data, y_data = fetch_openml("mnist_784", return_X_y=True)

onehotencoder = OneHotEncoder(categories="auto")

X_data = X_data.to_numpy().astype(numpy_dtype)
# X_data = X_data / (X_data.max() - X_data.min())
x_padding = 32 * (1 + X_data.shape[-1] // 32) - X_data.shape[-1]
np.pad(X_data, [(0, 0), (0, x_padding)], constant_values=0)
X_data = (X_data - X_data.mean()) / X_data.std()
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, stratify=y_data)

y_train = onehotencoder.fit_transform(y_train.to_numpy().reshape(-1, 1)).toarray().astype(numpy_dtype)
y_test = onehotencoder.fit_transform(y_test.to_numpy().reshape(-1, 1)).toarray().astype(numpy_dtype)

X_train_torch = torch.from_numpy(X_train).to(device)
X_test_torch = torch.from_numpy(X_test).to(device)
y_train_torch = torch.from_numpy(y_train).to(device)
y_test_torch = torch.from_numpy(y_test).to(device)

del X_data
del y_data

torch_chunk_size = int(32 * np.ceil(chunk_size / 32))
input_size = X_train.shape[1]
n_chunks = int(np.ceil(X_train.shape[0] / torch_chunk_size))

hidden_size = 1000
hidden_size = int(32 * np.ceil(hidden_size / 32))


def leaky_relu(x, negative_slope=0.01):
    return np.maximum(x, 0) + negative_slope * np.minimum(x, 0)


def relu(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return 0.5 + 0.5 * np.tanh(x / 2.0)


def tanh(x):
    return np.tanh(x)


def softplus(x, beta=1.0, threshold=20):
    return np.where(beta * x > threshold, x, np.log(1.0 + np.exp(np.array(beta * x, dtype=np.float128))) / beta).astype(x.dtype)


# act = torch.nn.functional.relu
# act = torch.nn.functional.sigmoid
# act = torch.nn.functional.tanh
# act = torch.nn.functional.leaky_relu
act = torch.nn.Softplus()

# np_act = relu
# np_act = sigmoid
# np_act = tanh
# np_act = leaky_relu
np_act = softplus


input_weights = np.random.normal(size=[input_size, hidden_size])
biases = np.random.normal(size=[hidden_size])

# output_weights = np.dot(np.linalg.pinv(hidden_nodes(X_train)), y_train)
output_weights = np.dot(imqrginv(np_act(np.dot(X_train, input_weights) + biases)), y_train)

prediction = np.dot(np_act(np.dot(X_test, input_weights) + biases), output_weights)
correct = 0
total = X_test.shape[0]

for i in range(total):
    predicted = np.argmax(prediction[i])
    actual = np.argmax(y_test[i])
    correct += 1 if predicted == actual else 0
accuracy = correct / total
print(f"NUMPY | Accuracy for {hidden_size} hidden nodes: {accuracy}")

del prediction
del predicted
del input_weights
del output_weights

mlgelm_learning_config = {**elm_learning_config, "weight_kappa": 0.05, "laplacian_lambda": 0.05, "k_neighbors": 16}
encoder_1 = MLGELMEncoder(
    encoder_depth=5,
    embedding_dim=256,
    activation=act,
    device=device,
    dtype=torch_dtype,
    elm_learning_config=mlgelm_learning_config,
)
# encoder_loss = encoder_1.learn_weights(torch.from_numpy(X_train).to(device=device, dtype=torch_dtype))

idx = np.arange(X_train.shape[0])

encoder_losses = []

for __ in range(1):
    np.random.shuffle(idx)
    for chunk in range(n_chunks):
        x = X_train_torch[idx[chunk * torch_chunk_size : min((chunk + 1) * torch_chunk_size, X_train.shape[0])]]
        # for _, (x, y) in enumerate(
        #     zip(
        #         torch.split(torch.from_numpy(X_train[idx]).to(device), torch_chunk_size),
        #         torch.split(torch.from_numpy(y_train[idx]).to(device), torch_chunk_size),
        #         strict=True,
        #     ),
        # ):
        print(f"Chunk {chunk}")

        t0 = time.process_time_ns()
        encoder_losses.append(encoder_1.learn_weights(x))
        t1 = time.process_time_ns()

        print(f"Train Time: {(t1 - t0) * 1e-9:.6f}s")
        print()

del x
# resnet_model = resnet34(weights=ResNet34_Weights.DEFAULT)
# resnet_model.eval()

# preprocess = ResNet34_Weights.DEFAULT.transforms()

# resnet_in = torch.from_numpy(X_test.reshape(-1, 28, 28))
# resnet_in = preprocess(torch.stack([resnet_in for _ in range(3)], dim=1))
# prediction = resnet_model(resnet_in).softmax(dim=0)

# for i in range(total):
#     predicted = np.argmax(prediction[i])
#     actual = np.argmax(y_test[i])
#     correct += 1 if predicted == actual else 0
# accuracy = correct / total
# print(f"ResNET | Accuracy for {hidden_size} hidden nodes: {accuracy}")

elm_hidden_weights_numpy = torch.empty((hidden_size, input_size), dtype=torch_dtype)
elm_hidden_bias_numpy = torch.empty((hidden_size), dtype=torch_dtype)
elm_output_weights_numpy = torch.zeros((y_train.shape[1], hidden_size), dtype=torch_dtype).numpy()

torch.nn.init.orthogonal_(elm_hidden_weights_numpy, gain=0.9)
elm_hidden_bias_numpy.unsqueeze_(0)
torch.nn.init.orthogonal_(elm_hidden_bias_numpy, gain=0.9)
elm_hidden_bias_numpy.squeeze_()
elm_hidden_bias_numpy.unsqueeze_(-1)
# torch.nn.init.normal_(elm_hidden_weights_numpy)
# torch.nn.init.normal_(elm_hidden_bias_numpy)

elm_hidden_weights_numpy = elm_hidden_weights_numpy.numpy()
elm_hidden_bias_numpy = elm_hidden_bias_numpy.numpy()

layers_0 = [
    ELMHiddenLinear(in_features=input_size, out_features=hidden_size, device=device, dtype=torch_dtype),
    ELMLinear(in_features=hidden_size, out_features=y_train.shape[1], device=device, dtype=torch_dtype),
]
# layers_1 = [
#     # ELMHiddenLinear(in_features=input_size, out_features=hidden_size, device=device, dtype=torch_dtype),
#     ELMHiddenLinear(in_features=input_size + y_train.shape[1], out_features=hidden_size, device=device, dtype=torch_dtype),
#     ELMLinear(in_features=hidden_size, out_features=y_train.shape[1], device=device, dtype=torch_dtype),
# ]
# layers_2 = [
#     # ELMHiddenLinear(in_features=input_size, out_features=hidden_size, device=device, dtype=torch_dtype),
#     ELMHiddenLinear(in_features=input_size + y_train.shape[1], out_features=hidden_size, device=device, dtype=torch_dtype),
#     ELMLinear(in_features=hidden_size, out_features=y_train.shape[1], device=device, dtype=torch_dtype),
# ]
# layers_3 = [
#     # ELMHiddenLinear(in_features=input_size, out_features=hidden_size, device=device, dtype=torch_dtype),
#     ELMHiddenLinear(in_features=input_size + y_train.shape[1], out_features=hidden_size, device=device, dtype=torch_dtype),
#     ELMLinear(in_features=hidden_size, out_features=y_train.shape[1], device=device, dtype=torch_dtype),
# ]
# layers_4 = [
#     # ELMHiddenLinear(in_features=input_size, out_features=hidden_size, device=device, dtype=torch_dtype),
#     ELMHiddenLinear(in_features=input_size + y_train.shape[1], out_features=hidden_size, device=device, dtype=torch_dtype),
#     ELMLinear(in_features=hidden_size, out_features=y_train.shape[1], device=device, dtype=torch_dtype),
# ]
model_0 = ELMSimpleSequential(*layers_0, elm_learning_config=elm_learning_config).to(device=device, dtype=torch_dtype)
# model_1 = ELMSimpleSequential(*layers_1, elm_learning_config=elm_learning_config).to(device=device, dtype=torch_dtype)
# model_2 = ELMSimpleSequential(*layers_2, elm_learning_config=elm_learning_config).to(device=device, dtype=torch_dtype)
# model_3 = ELMSimpleSequential(*layers_3, elm_learning_config=elm_learning_config).to(device=device, dtype=torch_dtype)
# model_4 = ELMSimpleSequential(*layers_4, elm_learning_config=elm_learning_config).to(device=device, dtype=torch_dtype)

# model_0(torch.rand(torch_chunk_size, X_train.shape[-1], dtype=torch_dtype, device=device))
# model_1(torch.rand(torch_chunk_size, X_train.shape[-1] + y_train.shape[-1], dtype=torch_dtype, device=device))
# model_2(torch.rand(torch_chunk_size, X_train.shape[-1] + y_train.shape[-1], dtype=torch_dtype, device=device))
# model_3(torch.rand(torch_chunk_size, X_train.shape[-1] + y_train.shape[-1], dtype=torch_dtype, device=device))
# model_4(torch.rand(torch_chunk_size, X_train.shape[-1] + y_train.shape[-1], dtype=torch_dtype, device=device))

A_inv = None
idx = np.arange(X_train.shape[0])
pred_times = []
learn_times = []
for __ in range(1):
    np.random.shuffle(idx)
    for _, (x, y) in enumerate(
        zip(
            torch.split(X_train_torch, torch_chunk_size),
            torch.split(y_train_torch, torch_chunk_size),
            strict=True,
        ),
    ):
        print(f"Chunk {_}")

        t0 = time.process_time_ns()
        # preds = model(x)
        preds_0 = model_0(x)
        t1 = time.process_time_ns()
        pred_times.append((t1 - t0) * 1e-9)
        print(f"Inference Time: {pred_times[-1]:.6f}s")

        # preds.append(preds_0)
        # # pred = torch.stack(preds, dim=-1).sum(dim=-1).argmax(dim=-1, keepdim=False)
        # pred = torch.stack(preds, dim=-1).argmax(dim=-1, keepdim=False).mode(dim=-1).values
        # pred = torch.nn.functional.one_hot(pred, y_test.shape[-1])

        # # preds_1 = model_1(torch.cat([x, preds_0], dim=-1))
        # preds_1 = model_1(torch.cat([x, pred], dim=-1))

        # preds.append(preds_1)
        # # pred = torch.stack(preds, dim=-1).sum(dim=-1).argmax(dim=-1, keepdim=False)
        # pred = torch.stack(preds, dim=-1).argmax(dim=-1, keepdim=False).mode(dim=-1).values
        # pred = torch.nn.functional.one_hot(pred, y_test.shape[-1])

        # # preds_2 = model_2(torch.cat([x, preds_1], dim=-1))
        # preds_2 = model_2(torch.cat([x, pred], dim=-1))

        # preds.append(preds_2)
        # # pred = torch.stack(preds, dim=-1).sum(dim=-1).argmax(dim=-1, keepdim=False)
        # pred = torch.stack(preds, dim=-1).argmax(dim=-1, keepdim=False).mode(dim=-1).values
        # pred = torch.nn.functional.one_hot(pred, y_test.shape[-1])

        # # preds_3 = model_3(torch.cat([x, preds_2], dim=-1))
        # preds_3 = model_3(torch.cat([x, pred], dim=-1))

        # preds.append(preds_3)
        # # pred = torch.stack(preds, dim=-1).sum(dim=-1).argmax(dim=-1, keepdim=False)
        # pred = torch.stack(preds, dim=-1).argmax(dim=-1, keepdim=False).mode(dim=-1).values
        # pred = torch.nn.functional.one_hot(pred, y_test.shape[-1])

        # # preds_4 = model_4(torch.cat([x, preds_3], dim=-1))
        # preds_4 = model_4(torch.cat([x, pred], dim=-1))

        t0 = time.process_time_ns()
        # model.learn_weights(targets=y, preds=None)
        model_0.learn_weights(targets=y, preds=None)
        t1 = time.process_time_ns()
        learn_times.append((t1 - t0) * 1e-9)

        # model_1.learn_weights(targets=y, preds=None)
        # model_2.learn_weights(targets=y, preds=None)
        # model_3.learn_weights(targets=y, preds=None)
        # model_4.learn_weights(targets=y, preds=None)
        print(f"Train Time: {learn_times[-1]:.6f}s")
        print()

x_test = torch.from_numpy(X_test).to(device=device, dtype=torch_dtype)
x_test_chunks = torch.split(x_test, torch_chunk_size)
predictions_0 = []
predictions_1 = []
predictions_2 = []
predictions_3 = []
predictions_4 = []
for test_chunk in x_test_chunks:
    original_batch = test_chunk.shape[0]
    padding = torch_chunk_size - test_chunk.shape[0]

    predictions = []

    # test_chunk_0 = torch.nn.functional.pad(test_chunk, pad=[0, 0, 0, padding], value=0)
    # predictions_0.append(model_0(test_chunk)[:original_batch])
    predictions_0.append(model_0(test_chunk))
    # prediction = act(torch.nn.functional.linear(test_chunk, elm_hidden_weights_tensor, elm_hidden_bias_tensor))
    # for idx in range(len(elm_output_weights_tensors)):
    #     pred_out = torch.nn.functional.linear(prediction, elm_output_weights_tensors[idx])
    #     match idx:
    #         case 0:
    #             predictions_0.append(pred_out)
    #         case 1:
    #             predictions_1.append(pred_out)
    #         case 2:
    #             predictions_2.append(pred_out)
    #         case 3:
    #             predictions_3.append(pred_out)
    #         case 4:
    #             predictions_4.append(pred_out)

    # predictions_0.append(model_0(test_chunk)[:original_batch])

    # predictions.append(predictions_0[-1])
    # # prediction = torch.stack(predictions, dim=-1).sum(dim=-1).argmax(dim=-1, keepdim=False)
    # prediction = torch.stack(predictions, dim=-1).argmax(dim=-1, keepdim=False).mode(dim=-1).values
    # prediction = torch.nn.functional.one_hot(prediction, y_test.shape[-1])

    # # test_chunk_1 = torch.nn.functional.pad(torch.cat([test_chunk, predictions_0[-1]], dim=-1), pad=[0, 0, 0, padding], value=0)
    # test_chunk_1 = torch.nn.functional.pad(torch.cat([test_chunk, prediction], dim=-1), pad=[0, 0, 0, padding], value=0)
    # predictions_1.append(model_1(test_chunk_1)[:original_batch])

    # predictions.append(predictions_1[-1])
    # # prediction = torch.stack(predictions, dim=-1).sum(dim=-1).argmax(dim=-1, keepdim=False)
    # prediction = torch.stack(predictions, dim=-1).argmax(dim=-1, keepdim=False).mode(dim=-1).values
    # prediction = torch.nn.functional.one_hot(prediction, y_test.shape[-1])

    # test_chunk_2 = torch.nn.functional.pad(torch.cat([test_chunk, predictions_1[-1]], dim=-1), pad=[0, 0, 0, padding], value=0)
    # # test_chunk_2 = torch.nn.functional.pad(torch.cat([test_chunk, prediction], dim=-1), pad=[0, 0, 0, padding], value=0)
    # predictions_2.append(model_2(test_chunk_2)[:original_batch])

    # predictions.append(predictions_2[-1])
    # # prediction = torch.stack(predictions, dim=-1).sum(dim=-1).argmax(dim=-1, keepdim=False)
    # prediction = torch.stack(predictions, dim=-1).argmax(dim=-1, keepdim=False).mode(dim=-1).values
    # prediction = torch.nn.functional.one_hot(prediction, y_test.shape[-1])

    # # test_chunk_3 = torch.nn.functional.pad(torch.cat([test_chunk, predictions_2[-1]], dim=-1), pad=[0, 0, 0, padding], value=0)
    # test_chunk_3 = torch.nn.functional.pad(torch.cat([test_chunk, prediction], dim=-1), pad=[0, 0, 0, padding], value=0)
    # predictions_3.append(model_3(test_chunk_3)[:original_batch])

    # predictions.append(predictions_3[-1])
    # # prediction = torch.stack(predictions, dim=-1).sum(dim=-1).argmax(dim=-1, keepdim=False)
    # prediction = torch.stack(predictions, dim=-1).argmax(dim=-1, keepdim=False).mode(dim=-1).values
    # prediction = torch.nn.functional.one_hot(prediction, y_test.shape[-1])

    # # test_chunk_4 = torch.nn.functional.pad(torch.cat([test_chunk, predictions_3[-1]], dim=-1), pad=[0, 0, 0, padding], value=0)
    # test_chunk_4 = torch.nn.functional.pad(torch.cat([test_chunk, prediction], dim=-1), pad=[0, 0, 0, padding], value=0)
    # predictions_4.append(model_4(test_chunk_4)[:original_batch])

# x_test = torch.nn.functional.pad(x_test, pad=[0, 0, 0, torch_chunk_size - X_test.shape[0]], value=0)
# prediction_0 = model(torch.from_numpy(X_test).to(device).to_padded_tensor(0, (torch_chunk_size, X_train.shape[-1])))
# prediction_0 = model(x_test)
prediction_0 = torch.cat(predictions_0)
# prediction_1 = torch.cat(predictions_1)
# prediction_2 = torch.cat(predictions_2)
# prediction_3 = torch.cat(predictions_3)
# prediction_4 = torch.cat(predictions_4)

correct = 0

prediction_0 = np.argmax(prediction_0.detach().cpu().numpy(), axis=-1)
# prediction_1 = np.argmax(prediction_1.detach().cpu().numpy(), axis=-1)
# prediction_2 = np.argmax(prediction_2.detach().cpu().numpy(), axis=-1)
# prediction_3 = np.argmax(prediction_3.detach().cpu().numpy(), axis=-1)
# prediction_4 = np.argmax(prediction_4.detach().cpu().numpy(), axis=-1)

predicted = scipy.stats.mode(
    np.array(
        [
            prediction_0,
            # prediction_1,
            # prediction_2,
            # prediction_3,
            # prediction_4,
        ],
    ),
    axis=0,
).mode
actual = np.argmax(y_test, axis=-1)
correct = (predicted == actual).sum()
accuracy = correct / total
print(f"TORCH CHUNKED TENSOR | Accuracy for {hidden_size} hidden nodes, gamma={gamma}, nets=11: {accuracy}")
# stats[(w, gamma)].append(accuracy)
