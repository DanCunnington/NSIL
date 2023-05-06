from src.NeuralSymbolicReasoner.base import NeuralSymbolicReasoner
from global_config import DEVICE, HYP_START_ID, HYP_END_ID
from tqdm import tqdm
import torch


class NeurASP(NeuralSymbolicReasoner):
    def __init__(self, args, logger, task):
        super().__init__(args, logger, task)
        self.NeurASP_data = task.data.NeurASP_data
        self.net_confs = self.task.net_confs
        self.im_to_net_map = self.task.image_to_net_map
        self.NeurASP_obj = None

        # Move networks to device
        for n in self.net_confs:
            self.net_confs[n].net.to(DEVICE)

    def _calculate_prob_of_model(self, m, net_confs, im_keys):
        prob = 1

        # Iterate over model possibilities
        for im_idx, im in enumerate(im_keys):
            num_out = self.net_confs[self.im_to_net_map[im]].num_out
            poss_net_outs = list(range(num_out))

            for net_out_idx, nn_out in enumerate(poss_net_outs):
                if m[im_idx] == nn_out:
                    prob = prob * net_confs[im_idx][net_out_idx]
        return prob

    def _calculate_grad(self, image_idx, image_str, models, probs, nn_confs):
        """
        Calculate gradients for the neural network
        @param image_idx: the image idx in models to calculate grads for
        @param image_str: the i1,i2,i3,.. str corresponding to the image idx
        @param models: all stable models for this example
        @param probs: model probabilities
        @param nn_confs: predicted neural network confidence scores
        @return: calculated gradients according to NeurASP semantic loss
        """
        gradients = []
        num_out = self.net_confs[self.im_to_net_map[image_str]].num_out
        poss_net_outs = list(range(num_out))

        # If there is only 1 stable model, we learn from complete interpretation
        if len(models) == 1:
            model = models[0]
            p = 0
            # For each possible network output, see which is in the model
            for net_out_idx, net_out in enumerate(poss_net_outs):
                if model[image_idx] == net_out:
                    gradients.append(1)
                    p = nn_confs[image_idx][net_out_idx]
                else:
                    gradients.append(-1)
            for net_out_idx, _ in enumerate(poss_net_outs):
                gradients[net_out_idx] = gradients[net_out_idx] / p
        # If there are more than 1 stable models, we use the equation in the proposition in the NeurASP paper
        else:
            denominator = sum(probs)
            for net_out_idx, net_out in enumerate(poss_net_outs):
                numerator = 0
                for m_idx, m in enumerate(models):
                    if m[image_idx] == net_out:
                        numerator += probs[m_idx] / nn_confs[image_idx][net_out_idx]
                    else:
                        for atom_idx, atom in enumerate(poss_net_outs):
                            if m[image_idx] == atom:
                                numerator -= probs[m_idx] / nn_confs[image_idx][atom_idx]
                if denominator == 0:
                    gradients.append(0)
                else:
                    gradients.append(numerator / denominator)
        return gradients

    def train(self, h):
        # ASP program
        # Wrap hypothesis in identifiers so we can extract it if needs be
        asp_program = f'{self.task.bk}\n{HYP_START_ID}\n{h}\n{HYP_END_ID}\n'
        data_list = self.NeurASP_data[0]
        obs_list = self.NeurASP_data[1]

        for e in range(self.args.num_net_epochs):
            for idx, data in enumerate(tqdm(data_list, ncols=50)):
                # Get the neural network output and initialise the gradients
                nn_output = []
                # Copy NN output without computation graph
                _nn_output = []
                nn_grads = []

                # If net_max_example_len passed, check it
                if not hasattr(self.args, 'net_max_example_len') or len(data) <= self.args.net_max_example_len:
                    for im in data:
                        im_net = self.net_confs[self.im_to_net_map[im]].net
                        im_data = data[im].to(DEVICE)
                        im_data = im_data.unsqueeze(0)
                        im_net_out = im_net(im_data)
                        im_net_out = torch.clamp(im_net_out, min=10e-8, max=1. - 10e-8)
                        nn_output.append(im_net_out)
                        _nn_output.append(im_net_out.view(-1).tolist())
                        nn_grads.append([0]*im_net_out.shape[0])

                    # Compute stable models
                    models = self.task.compute_stable_models(asp_program, obs_list[idx], len(data), for_ilp=False)
                    if len(models) > 0:
                        # Calculate probability of each model
                        probs = [self._calculate_prob_of_model(m, _nn_output, list(data.keys())) for m in models]

                        # Calculate NN gradients
                        nn_grads = [self._calculate_grad(im_idx, im, models, probs, _nn_output)
                                    for im_idx, im in enumerate(data)]

                        # Perform backpropagation
                        for nn_out_idx in range(len(nn_output)):
                            grad_tensor = torch.FloatTensor(nn_grads[nn_out_idx]).unsqueeze(0).to(DEVICE) * -1
                            nn_output[nn_out_idx].backward(grad_tensor, retain_graph=True)

                        # Update optimisers
                        for net_name in self.net_confs:
                            self.net_confs[net_name].optim.step()
                            self.net_confs[net_name].optim.zero_grad()


