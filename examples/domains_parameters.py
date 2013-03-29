# fill in paths to envelopes
envelopes = ['path to some envelope file',
             'path to another envelope file']

# if True, every time a new envelope is loaded the domains are reset to
# random numbers. if False, the converged domains are kept, which lets
# a pattern be evolved between configurations
reset_each = False

# these pertain to domains_example_2 only. "kicking" adds noise to the
# speckle so that the pattern reconverges at a similar but not exact
# configuration. kicks is the number of times the domains are converged
# and kicked. kick_amount is a sort of decorrelation parameter, with
# higher amounts leading to more rapid decorrelation. experimental.
kicks = 500
kick_amount = 0.3


