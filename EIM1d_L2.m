%% EIM for ReLU_k on [0,1] in the L2 norm, k = 1,...,4

iter = 200;

b = linspace(-2, 2, 1000)';
xset = linspace(0,1,1000)';

w = [-1, 1];
[W, Bb] = meshgrid(w, b);
muset = [W(:) Bb(:)];

musize = size(muset,1);
xsize = size(xset,1);

error = zeros(iter,4);
Lambda = error;
for k=1:4
    f = @(z, b) max(0, b(1).*z+b(2)).^k;
    muid = zeros(iter,1);
    xid = muid;
    L = zeros(musize,1);
    B = zeros(iter,iter);
    Q = zeros(xsize,iter);

    fx = zeros(xsize,musize);
    for i = 1:musize
        fx(:,i) = f(xset, muset(i, :));
    end

    for n = 1:iter
        if n == 1
            L = h*sum(fx.*fx);
        else
            r = fx - Q(:, 1:n-1)*( B(1:n-1, 1:n-1)\fx(xid(1:n-1),:) );
            L = h*sum(r.*r);
        end
        [~, muid(n)] = max(L);
        if n == 1
            r = fx(:,muid(n));
        else
            r = fx(:,muid(n)) - Q(:,1:n-1)*( B(1:n-1,1:n-1)\fx(xid(1:n-1),muid(n)) );
        end
        [~, xid(n)] = max(abs(r));
        error(n,k) = sqrt(h*sum(r.*r));

        Q(:, n) = r/r(xid(n));
        B(n, 1:n) = Q(xid(n), 1:n);
        Lambda(n) = norm(Q,'inf');
        if mod(n,50) == 0
            fprintf('EIM at the %d-th step, k = %d, error is %e\n',n, k, error(n,k));
        end
    end

    mu = muset(muid,:);
    x = xset(xid,:);
end

% avgLambda = Lambda;
% h = zeros(iter,1);
% for i=1:iter
%     avgLambda(i) = (prod(1+Lambda(1:i))).^(1/i);
%     idx = (mu(1:i,1)==1);
%     node = sort(mu(idx,2));
%     if length(node)==1
%         h(i)=1;
%     else
%         h(i) = min(node(2:end)-node(1:end-1));
%     end
% end

%% Compute and plot the order of convergence of the EIM for ReLUk

N = 1:iter;

temp = polyfit(log(N(50:end)),log(error(50:end,1)),1);
fprintf('Order of EIM error for ReLU1 in L2 is %.3e \n', -temp(1));
loglog(N,error(:,1),'o','MarkerSize',5);
hold on

temp = polyfit(log(N(50:end)),log(error(50:end,2)),1);
fprintf('Order of EIM error for ReLU2 in L2 is %.3e \n', -temp(1));
loglog(N,error(:,2),'*','MarkerSize',5);

temp = polyfit(log(N(50:end)),log(error(50:end,3)),1);
fprintf('Order of EIM error for ReLU3 in L2 is %.3e \n', -temp(1));
loglog(N,error(:,3),'d','MarkerSize',5);

temp = polyfit(log(N(40:end)),log(error(40:end,4)),1);
fprintf('Order of EIM error for ReLU4 in L2 is %.3e \n', -temp(1));
loglog(N,error(:,4),'s','MarkerSize',5);

loglog(N,N.^(-1.5),'-.',N,N.^(-2.5),'-.',N,N.^(-3.5),'-.',N,N.^(-4.5),'-.');
