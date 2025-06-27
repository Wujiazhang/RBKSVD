% test.m
rng(42); % 设置随机种子保证可重复性
m = 500; n = 200;  % 矩阵维度
k = 10;            % 目标秩
bsize = k;         % 块大小
max_iter = 30;     % 最大迭代次数

% 生成矩阵A (奇异值缓慢衰减)
[U0, ~] = qr(randn(m, min(m,n)), 0);
[V0, ~] = qr(randn(n, min(m,n)), 0);
s = 1./(1 + 0.05*(0:min(m,n)-1)); % 缓慢衰减的奇异值
%s = 1.5.^linspace(-40, 40,min(m,n));
S0 = diag(s);
A = U0 * S0 * V0';
% 预分配存储空间
eps_fro_bk = zeros(max_iter, 1); eps_fro_si = zeros(max_iter, 1);   % Frobenius范数ε
eps_spec_bk = zeros(max_iter, 1); eps_spec_si = zeros(max_iter, 1); % 谱范数ε
eps_pv_bk = zeros(max_iter, 1); eps_pv_si = zeros(max_iter, 1);    % 逐向量ε

% 计算真实SVD作为基准
[U_true, S_true, V_true] = svd(A, 'econ');
A_true_k = U_true(:,1:k) * S_true(1:k,1:k) * V_true(:,1:k)';
true_svals = diag(S_true);
sigma_kplus1 = true_svals(k+1); % 第k+1个奇异值

% 最优误差
optimal_fro = norm(A - A_true_k, 'fro');
optimal_spec = norm(A - A_true_k, 2);

% 主循环：测试不同迭代次数
for iter = 1:max_iter
    % 运行BK-SVD
    [Ubk, Sbk, Vbk] = bksvd(A, k, iter, bsize, false);
    Abk = Ubk * Sbk * Vbk';
    
    % 计算BK-SVD的三种ε
    diff_fro_bk = norm(A - Abk, 'fro');
    eps_fro_bk(iter) = max((diff_fro_bk / optimal_fro) - 1,1e-16);
    
    diff_spec_bk = norm(A - Abk, 2);
    eps_spec_bk(iter) = max((diff_spec_bk / optimal_spec) - 1,1e-16);
    
    % 逐向量PCA误差
    pv_error = 0;
    for i = 1:k
        u_true = U_true(:, i);
        u_approx = Ubk(:, i);
        error_i = abs(u_true'*(A*A')*u_true - u_approx'*(A*A')*u_approx);
        pv_error = max(pv_error, error_i);
    end
    eps_pv_bk(iter) = pv_error / (sigma_kplus1^2);
    
    % 运行SI-SVD
    [Usi, Ssi, Vsi] = sisvd(A, k, iter, bsize, false);
    Asi = Usi * Ssi * Vsi';
    
    % 计算SI-SVD的三种ε
    diff_fro_si = norm(A - Asi, 'fro');
    eps_fro_si(iter) = max((diff_fro_si / optimal_fro) - 1,1e-16);
    
    diff_spec_si = norm(A - Asi, 2);
    eps_spec_si(iter) = max((diff_spec_si / optimal_spec) - 1,1e-16);
    
    % 逐向量PCA误差
    pv_error = 0;
    for i = 1:k
        u_true = U_true(:, i);
        u_approx = Usi(:, i);
        error_i = abs(u_true'*(A*A')*u_true - u_approx'*(A*A')*u_approx);
        pv_error = max(pv_error, error_i);
    end
    eps_pv_si(iter) = pv_error / (sigma_kplus1^2);
    
    fprintf('Iteration %d complete\n', iter);
end

% 绘制结果（使用对数刻度）
figure('Position', [100, 100, 1200, 400]);

% Frobenius范数ε
subplot(1, 3, 1);
semilogy(1:max_iter, eps_fro_bk, 'b-o', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
hold on;
semilogy(1:max_iter, eps_fro_si, 'r--s', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
title('Frobenius Norm ε');
xlabel('Iterations');
ylabel('ε (log scale)');
grid on;
yline(1e-16, 'k--', 'LineWidth', 1); % 机器精度参考线
legend('Block Krylov', 'Simult. Iter.','1e-16', 'Location', 'best');
% 谱范数ε
subplot(1, 3, 2);
semilogy(1:max_iter, eps_spec_bk, 'b-o', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
hold on;
semilogy(1:max_iter, eps_spec_si, 'r--s', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
title('Spectral Norm ε');
xlabel('Iterations');
ylabel('$$\epsilon (log scale)$$', 'Interpreter', 'latex');
grid on;
yline(1e-16, 'k--', 'LineWidth', 1); % 机器精度参考线
legend('Block Krylov', 'Simult. Iter.','1e-16', 'Location', 'best');

% 逐向量PCA ε
subplot(1, 3, 3);
semilogy(1:max_iter, eps_pv_bk, 'b-o', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
hold on;
semilogy(1:max_iter, eps_pv_si, 'r--s', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
title('Per Vector PCA ε');
xlabel('Iterations');
ylabel('ε (log scale)');
grid on;
yline(1e-16, 'k--', 'LineWidth', 1); % 机器精度参考线
legend('Block Krylov', 'Simult. Iter.','1e-16', 'Location', 'best');
saveas(gcf, 'log_error.png');
%sgtitle('Theoretical Guarantee ε Values (Theorem 3.1)');

% 添加参考线说明
% annotation('textbox', [0.4, 0.01, 0.2, 0.05], 'String', ...
%     'Black dashed line: Machine precision (~1e-15)', ...
%     'EdgeColor', 'none', 'HorizontalAlignment', 'center');