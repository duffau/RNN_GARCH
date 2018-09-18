# # Plotting absolut returns and estimated sigma_t
# # -----------------------------------------------
# # Evaluating model after minimization of loss function and saving in DataFrame for plotting
# df_train['sigma2_est'] = model.log_likelihood(y=df_train['return_dm'])
#
# plt.subplot(211)
# df_train['return'].plot()
#
# plt.subplot(212)
# np.sqrt(df_train['return'] ** 2).plot()
# np.sqrt(df_train['sigma2_opt']).plot()
# plt.show()