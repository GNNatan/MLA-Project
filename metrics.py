import torch

from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from MIL import AttentionMIL

from training import MultiBagMILDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_metrics(model, loader):
    model.eval()

    probs = []
    preds = []
    labels = []

    for data, label in tqdm(loader):
        data = data.to(device)
        label = label.cpu().numpy()

        Y_prob, Y_pred, _ = model(data)

        prob = Y_prob.detach().cpu().numpy().flatten()[0]
        pred = int(Y_pred)

        probs.append(prob)
        preds.append(pred)
        labels.append(int(label))
    
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    auc = roc_auc_score(labels, probs)

    return accuracy, precision, recall, f1, auc

    
        

def main():
    test_set = [str(i) for i in range(17, 25)]
    test_data = MultiBagMILDataset(test_set)

    test_loader = torch.utils.data.DataLoader(test_data, num_workers=16)

    checkpoint_name = "checkpoints/attention/latest.pth"
    checkpoint = torch.load(checkpoint_name, map_location=device)

    model = AttentionMIL("attention")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    accuracy, precision, recall, f1, auc = calculate_metrics(model, test_loader)
    
    with open("metrics.txt", 'w') as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F-Score: {f1}\n")
        f.write(f"AUC: {auc}\n")

if __name__ == "__main__":
    main()
