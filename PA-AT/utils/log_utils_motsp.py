def log_values(oa, ob, oc, od, grad_norms, epoch, batch_id, step,
               reinforce_loss, bl_loss, tb_logger, opts):
    
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
   

    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))

    # Log values to tensorboard
    if not opts.no_tensorboard:
        tb_logger.log_value('obj1 for pref 0', oa[0], step)
        tb_logger.log_value('obj2 for pref 0', oa[1], step)
        tb_logger.log_value('obj3 for pref 0', oa[2], step)
        tb_logger.log_value('cstr for pref 0', oa[3], step)
        tb_logger.log_value('nll for pref 0', oa[4] , step)

        tb_logger.log_value('obj1 for pref 4', ob[0], step)
        tb_logger.log_value('obj2 for pref 4', ob[1], step)
        tb_logger.log_value('obj3 for pref 4', ob[2], step)
        tb_logger.log_value('cstr for pref 4', ob[3], step)
        tb_logger.log_value('nll for pref 4', ob[4] , step)

        tb_logger.log_value('obj1 for pref 8', oc[0], step)
        tb_logger.log_value('obj2 for pref 8', oc[1], step)
        tb_logger.log_value('obj3 for pref 8', oc[2], step)
        tb_logger.log_value('cstr for pref 8', oc[3], step)
        tb_logger.log_value('nll for pref 8', oc[4] , step)

        tb_logger.log_value('obj1 for pref 15', od[0], step)
        tb_logger.log_value('obj2 for pref 15', od[1], step)
        tb_logger.log_value('obj3 for pref 15', od[2], step)
        tb_logger.log_value('cstr for pref 15', od[3], step)
        tb_logger.log_value('nll for pref 15', od[4] , step)

        tb_logger.log_value('actor_loss', reinforce_loss.item(), step)
        

        tb_logger.log_value('grad_norm', grad_norms[0], step)
        tb_logger.log_value('grad_norm_clipped', grad_norms_clipped[0], step)

        if opts.baseline == 'critic':
            tb_logger.log_value('critic_loss', bl_loss.item(), step)
            tb_logger.log_value('critic_grad_norm', grad_norms[1], step)
            tb_logger.log_value('critic_grad_norm_clipped', grad_norms_clipped[1], step)
