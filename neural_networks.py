class NeuralNetwork:
    # What:
    # Highlight the two main “modes” of machine learning.
    # Simplified examples of how neural networks work
    #   Supervised learning
    #       What it shows: A tiny neural network that learns by comparing predictions to known outputs (labels).
    #       Key mechanism:
    #           Forward pass: weighted sum → sigmoid activation.
    #           Backward pass: error → gradient → weight updates.
    #       Most important concept:
    #           Sigmoid derivative — because it’s the slope that tells the network how much to adjust when learning.
    #       Analogy: Like practicing math problems with an answer key — you know the right outputs, and you adjust until your guesses match.
    #   Unsupervised learning
    #       What it shows: Grouping points into clusters without knowing the outputs (no labels).
    #       Key mechanism:
    #           Assign points to nearest center.
    #           Update centers to the average of their points.
    #           Repeat until stable.
    #       Most important concept:
    #           K-means clustering — because it’s the simplest way to see how unsupervised learning organizes data by similarity.
    #       Analogy: Like sorting puzzle pieces by color/shape without knowing the final picture — the groups emerge naturally.
    # ---------------------------------------------------------------------------------------------------------------------------
    # Terms:
    # Weight: An adjustable number multiplying an input, telling the model how important that input is
    # Bias: An adjustable number added after the weighted sum, shifting the output up or down
    # Activation function: A function applied to the linear output to add non-linearity. Sigmoid is a common activation function that maps to [0, 1]
    # Sigmoid derivative: The rate at which sigmoid changes; for output s, it’s s(1 - s)
    #   Important for understanding supervised learning
    # Loss function: A measure of error. Lower is better
    # Gradient: The "slope" of the loss with respect to a parameter; tells us how to change that parameter to reduce loss
    # Learning rate: A knob that sets how big the parameter updates are
    # Epoch: One full pass through the dataset
    # K-means clustering is an unsupervised learning algorithm.
    #   Important for understanding unsupervised learning
    #   It groups data points into k clusters (k = number of groups you choose).
    #   Each cluster has a center (called a centroid).
    #   Points are assigned to the cluster whose center they’re closest to.
    #   The centers are updated until things settle down.
    #   * Pick k (number of clusters): Example: k=2 means "split the data into 2 groups" - Pick 2 kids as “leaders” - initial centers
    #   * Choose starting centers: Pick k random points as initial guesses for the cluster centers
    #   * Assign points: For each data point, calculate its distance to each center. Put the point in the cluster of the nearest center - Each kid joins the group of the leader they’re closest to
    #   * Update centers: For each cluster, move its center to the average position of the points inside it - Each leader moves to the middle of their group (average position)
    #   * Repeat: Keep re-assigning points and updating centers until the centers stop moving much - Repeat until the groups stop changing

    def __init__(self, layers):
        self.layers = layers

    @staticmethod
    def supervised_and():
        # What:
        # The code is a tiny supervised neural network that learns the AND function by repeatedly guessing,
        # checking against the correct answers, and adjusting itself to improve.
        # ----------------------------------------
        # What to observe:
        # Early loss: Higher at first because weights and bias start at zero
        # Later loss: Decreases as the model learns
        # Predictions: After training, outputs should be close to 0 for [0,0], [0,1], [1,0], and close to 1 for [1,1]
        # ----------------------------------------
        # Goal
        # AND function: Takes two inputs (0 or 1) and returns 1 only if both are 1, otherwise 0.
        # Inputs (X): The four combinations [0,0], [0,1], [1,0], [1,1], [0,1], [1,1].
        # Targets (y): The correct outputs [0, 0, 0, 1, 0, 1].
        # ----------------------------------------

        # --- Data: AND truth table ---
        X = [
            [0, 0],  # input 1, input 2
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 1],
            [1, 1]
        ]
        y = [0, 0, 0, 1, 0, 1]  # AND outputs (labels)
        # y = [0, 1, 0, 1, 0, 0]  # Experiment: wrong output

        # --- Helper functions ---
        # Squashes any number into a value between 0 and 1.
        # It’s shaped like a smooth “S” curve, starting near 0 for very negative inputs, rising through 0.5 at input 0, and approaching 1 for very positive inputs
        # y
        # 1.0 |                              ________
        #     |                           _/
        # 0.8 |                         _/
        #     |                      _/
        # 0.6 |                    _/
        #     |                 _/
        # 0.5 |-----------------*----------------------  (x=0 crosses y=0.5)
        #     |             _/
        # 0.4 |           _/
        #     |        _/
        # 0.2 |      _/
        #     |    /
        # 0.0 |___/___________________________________ x
        #        -6   -4   -2    0    2    4    6

        def sigmoid(z):
            # Squashes any real number to a value between 0 and 1
            e = 2.718281828459045
            return 1 / (1 + (e ** (-z)))  # e^(-z)

        def sigmoid_derivative(s):
            # Derivative of sigmoid with respect to its output s
            return s * (1 - s)

        # --- Initialize parameters ---
        w1, w2 = 0.0, 0.0   # weights (adjustable multipliers for each input)
        b = 0.0             # bias (adjustable baseline)
        lr = 0.1            # learning rate: Controls the step size - how big a step we take
        epochs = 3000       # how many training iterations

        # --- Training loop ---
        print("\nSUPERVISED AND\n")
        for epoch in range(epochs):
            total_loss = 0.0
            for i in range(len(X)):
                x1, x2 = X[i]
                target = y[i]

                # 1) Forward pass: linear -> activation
                # Weights (w1, w2): Adjustable multipliers for each input. They tell the model how strongly each input affects the output.
                # Bias (b): An adjustable baseline added to the weighted sum. It shifts the decision threshold up or down.
                z = w1 * x1 + w2 * x2 + b         # linear combination
                # Squashes z into a number between 0 and 1 so we can interpret it like a probability
                # Apply sigmoid to get the prediction
                # If z is large and positive, sigmoid is near 1; if z
                # is very negative, sigmoid is near 0; around 0, it’s near 0.5.
                pred = sigmoid(z)                 # activation

                # 2) Loss (measuring “wrongness” - mean squared error for this one example)
                # a single number telling us how far off our prediction is. Training tries to reduce this number.
                # Mean squared error (MSE): loss = (pred - target)^2
                error = pred - target
                loss = error * error
                total_loss += loss

                # 3) Backward pass: compute gradients (simple chain rule)
                # "Fixes mistakes" - Figure out how to nudge w1, w2, and b to reduce loss
                # Chain rule - We combine small derivatives:
                d_loss_d_pred = 2 * error                 # d(MSE)/d(pred) - loss to prediction
                d_pred_d_z = sigmoid_derivative(pred)     # d(sigmoid)/d(z) - prediction to z, which is a derivative of sigmoid
                d_loss_d_z = d_loss_d_pred * d_pred_d_z   # d(loss)/d(z) - combine loss and z

                # Gradients for weights and bias
                # The direction and amount to change a parameter to reduce the loss
                d_loss_d_w1 = d_loss_d_z * x1
                d_loss_d_w2 = d_loss_d_z * x2
                d_loss_d_b  = d_loss_d_z

                # 4) Update parameters (gradient descent)
                # Update rule:
                # Weights (w) = w - lr * gradient
                # Bias (b) = b - lr * gradient
                w1 = w1 - lr * d_loss_d_w1
                w2 = w2 - lr * d_loss_d_w2
                b  = b - lr * d_loss_d_b

            # Print status occasionally
            if epoch % 500 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(X):.4f}")

        # --- Test after training ---
        print("\nTrained parameters:")
        print(f"w1={w1:.3f}, w2={w2:.3f}, b={b:.3f}")

        print("\nPredictions:")
        # Forward -> loss -> backward -> update
        # Tiny improvements compound; the parameters move towards values that make predictions match targets.
        for x1, x2 in X:
            z = w1 * x1 + w2 * x2 + b
            pred = sigmoid(z)
            print(f"Input [{x1},{x2}] -> {pred:.3f}")

    @staticmethod
    def unsupervised_xor():
        # ----------------------------------------
        # What:
        # We group the XOR inputs into two clusters WITHOUT knowing the correct outputs.
        # The algorithm repeatedly:
        #   - guesses where the "centers" of two groups are,
        #   - assigns each point to its nearest center,
        #   - moves each center to the average position of its assigned points,
        # and improves these guesses over a few rounds.
        # then a simple post-step to map clusters to XOR labels (0/1) by majority vote
        # ----------------------------------------
        # What to observe:
        # Early centers: Start as rough guesses.
        # Assignments: Points [0,0] and [1,1] tend to group together; [0,1] and [1,0] group together.
        # Centers stabilize: After a few iterations, centers stop changing much.
        # Since we are working with XOR, k-means clustering naturally groups [0,0] with [1,1] and [0,1] with [1,0] — because those pairs are closest together
        # After mapping clusters to labels by majority vote, the final outputs match XOR: 0, 1, 1, 0.
        # ----------------------------------------
        # Goal
        # Unsupervised learning on XOR inputs:
        # Inputs (points): [0,0], [0,1], [1,0], [1,1]
        # Clusters: Two groups found by the algorithm (k=2), without an answer key.
        # Final output: [0, 1, 1, 0]
        # ----------------------------------------

        # --- Data: XOR points (no outputs provided) ---
        points = [[0, 0], [0, 1], [1, 0], [1, 1]]

        # --- Helper functions ---
        def distance(p, c):
            # Euclidean distance (straight-line distance) between a point p and a center c
            # distance = sqrt( (px - cx)^2 + (py - cy)^2 )
            dx = p[0] - c[0]
            dy = p[1] - c[1]
            return (dx * dx + dy * dy) ** 0.5

        def mean_of(points_list):
            # Average position (center of mass) of points in a cluster
            # If the cluster is empty, we return None so we can handle it safely.
            if not points_list:
                return None
            xs = [p[0] for p in points_list]
            ys = [p[1] for p in points_list]
            return [sum(xs) / len(xs), sum(ys) / len(ys)]

        # --- Initialize centers (starting guesses) ---
        # We pick two points as initial centers. This is arbitrary; different starts may lead to the same final clusters.
        centers = [[0, 0], [1, 1]]

        # --- K-means loop: assign points, update centers, repeat ---
        print("\nUNSUPERVISED XOR CLUSTERING (k-means)\n")
        epochs = 5  # a few iterations are enough for 4 points
        for epoch in range(epochs):
            # Step 1: Assign each point to the nearest center
            clusters = {0: [], 1: []}
            for p in points:
                d0 = distance(p, centers[0])
                d1 = distance(p, centers[1])
                # Nearest-center rule: point goes to the cluster with smaller distance
                if d0 <= d1:
                    clusters[0].append(p)
                else:
                    clusters[1].append(p)

            # Step 2: Update each center to the mean of its assigned points
            new_centers = centers[:]  # copy
            for k in [0, 1]:
                m = mean_of(clusters[k])
                # If the cluster isn't empty, move the center to the mean;
                # if empty, keep the old center to avoid losing a cluster.
                if m is not None:
                    new_centers[k] = m

            centers = new_centers

            print(f"Epoch {epoch}:")
            print(f"  Cluster 0 points: {clusters[0]}  -> center: {centers[0]}")
            print(f"  Cluster 1 points: {clusters[1]}  -> center: {centers[1]}")

        # --- Assignments ---
        print("\nAssignment (these are not labels, just individual clusters):")
        for p in points:
            d0 = distance(p, centers[0])
            d1 = distance(p, centers[1])
            label = 0 if d0 <= d1 else 1
            print(f"Point {p} -> Cluster {label}")

        # --- Helper: true XOR of a point for label mapping ---
        def xor_label(p):
            # XOR is 1 if exactly one of x1, x2 is 1; otherwise 0
            return 1 if (p[0] + p[1]) == 1 else 0

        # --- Rebuild clusters once more using final centers (for mapping) ---
        final_clusters = {0: [], 1: []}
        for p in points:
            d0 = distance(p, centers[0])
            d1 = distance(p, centers[1])
            (final_clusters[0] if d0 <= d1 else final_clusters[1]).append(p)

        print("\nFinal clusters (for mapping):", final_clusters)

        # --- Map each cluster to XOR label by majority vote ---
        cluster_to_label = {}
        for k in [0, 1]:
            truths = [xor_label(p) for p in final_clusters[k]]
            ones = sum(truths)
            zeros = len(truths) - ones
            cluster_to_label[k] = 1 if ones > zeros else 0

        print("Cluster → XOR label mapping:", cluster_to_label)

        # --- Print outputs aligned to XOR truth table ---
        print("\nFinal outputs aligned to XOR:")
        for p in points:
            d0 = distance(p, centers[0])
            d1 = distance(p, centers[1])
            cid = 0 if d0 <= d1 else 1
            out = cluster_to_label[cid]
            print(f"Point {p} -> Output {out}")

NeuralNetwork.supervised_and()
print("\n------------------------------------")
NeuralNetwork.unsupervised_xor()
