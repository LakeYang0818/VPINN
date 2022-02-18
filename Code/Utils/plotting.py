import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

    # =========================================================================
    #     Plotting
    # =========================================================================
    x_quad_plot = X_quad_train
    y_quad_plot = np.empty(len(x_quad_plot))
    y_quad_plot.fill(1)

    x_train_plot = X_u_train
    y_train_plot = np.empty(len(x_train_plot))
    y_train_plot.fill(1)

    x_f_plot = X_f_train
    y_f_plot = np.empty(len(x_f_plot))
    y_f_plot.fill(1)

    fig = plt.figure(0)
    gridspec.GridSpec(3,1)

    plt.subplot2grid((3,1), (0,0))
    plt.tight_layout()
    plt.locator_params(axis='x', nbins=6)
    plt.yticks([])
    plt.title('$Quadrature \,\, Points$')
    plt.xlabel('$x$')
    plt.axhline(1, linewidth=1, linestyle='-', color='red')
    plt.axvline(-1, linewidth=1, linestyle='--', color='red')
    plt.axvline(1, linewidth=1, linestyle='--', color='red')
    plt.scatter(x_quad_plot,y_quad_plot, color='green')

    plt.subplot2grid((3,1), (1,0))
    plt.tight_layout()
    plt.locator_params(axis='x', nbins=6)
    plt.yticks([])
    plt.title('$Training \,\, Points$')
    plt.xlabel('$x$')
    plt.axhline(1, linewidth=1, linestyle='-', color='red')
    plt.axvline(-1, linewidth=1, linestyle='--', color='red')
    plt.axvline(1, linewidth=1, linestyle='--', color='red')
    plt.scatter(x_train_plot,y_train_plot, color='blue')

    fig.tight_layout()
    fig.set_size_inches(w=10,h=7)
    plt.savefig('Results/Train-Quad-pnts.pdf')
    #++++++++++++++++++++++++++++

    font = 24

    fig, ax = plt.subplots()
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
    plt.xlabel('$iteration$', fontsize = font)
    plt.ylabel('$loss \,\, values$', fontsize = font)
    plt.yscale('log')
    plt.grid(True)
    iteration = [total_record[i][0] for i in range(len(total_record))]
    loss_his  = [total_record[i][1] for i in range(len(total_record))]
    plt.plot(iteration, loss_his, 'gray')
    plt.tick_params( labelsize = 20)
    fig.set_size_inches(w=11,h=5.5)
    plt.savefig('Results/loss_'+str(np.around(k, 3))+'_.pdf')
    #++++++++++++++++++++++++++++

    pnt_skip = 25
    fig, ax = plt.subplots()
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=8)
    plt.xlabel('$x$', fontsize = font)
    plt.ylabel('$u$', fontsize = font)
    plt.axhline(0, linewidth=0.8, linestyle='-', color='gray')
    for xc in grid:
        plt.axvline(x=xc, linewidth=2, ls = '--')
    plt.plot(X_test, u_test, linewidth=1, color='r', label=''.join(['$exact$']))
    plt.plot(X_test[0::pnt_skip], u_pred[0::pnt_skip], 'k*', label='$VPINN$')
    plt.tick_params( labelsize = 20)
    legend = plt.legend(shadow=True, loc='upper left', fontsize=18, ncol = 1)
    fig.set_size_inches(w=11,h=5.5)
    plt.savefig('Results/prediction_'+str(np.around(k, 3))+'_.pdf')
    #++++++++++++++++++++++++++++

    fig, ax = plt.subplots()
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=8)
    plt.xlabel('$x$', fontsize = font)
    plt.ylabel('point-wise error', fontsize = font)
    plt.yscale('log')
    plt.axhline(0, linewidth=0.8, linestyle='-', color='gray')
    for xc in grid:
        plt.axvline(x=xc, linewidth=2, ls = '--')
    plt.plot(X_test, abs(u_test - u_pred), 'k')
    plt.tick_params( labelsize = 20)
    fig.set_size_inches(w=11,h=5.5)
    plt.savefig('Results/error_'+str(np.around(k, 3))+'_.pdf')
    #++++++++++++++++++++++++++++
