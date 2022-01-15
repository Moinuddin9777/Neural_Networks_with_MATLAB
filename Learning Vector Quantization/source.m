t = ind2vec(c);
% LVQ network represents clusters of vectors with hidden neurons, and groups the clusters with output neurons to form the desired classes 
colormap(hsv);
plotvec(x,c)
title(‘Input Vectors’);
xlabel(‘x(1)’);
ylabel(‘x(2)’);

% network is then configured for inputs X and targets T
net = lvqnet(4,0.1);
net = configure(net,x,t);

% Train the network:
net.trainParam.epochs=150;
net=train(net,x,t);

cla;
plotvec(x,c);
hold on;
plotvec(net.IW{1}’,vec2ind(net.LW{2}),’o’);

