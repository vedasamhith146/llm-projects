Okay so first what i have did in this project is that i have considered four cases 
(i) without residuals and normalizations
(ii) with residuals without normalizations
(iii) without residuals with normalizations
(iv) With residuals and normalizations
So the main aim was to observe the per-layer activation norms and per-layer gradient norms (We require these two to be healthy for the model to learn well.) I have made an observation for every 10 steps upto 200 steps. 
In the first case, no residuals and no normalizations case the activations will die out after first 2-3 layers (You may observe that the last layer's activations are finite while the before layers have zero activations how is this even possible? but remember about the weight-tying scheme which we use in transformer based language models. So after the last layer we have the same token embedding table that is at the same table at the start, which we call as Language modeling head.)
