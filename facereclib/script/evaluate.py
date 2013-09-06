#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <manuel.guenther@idiap.ch>
# Tue Jul 2 14:52:49 CEST 2013
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""This script evaluates the given score files and computes EER, HTER.
It also is able to plot CMC and ROC curves."""

from .. import utils
import argparse
import bob
import numpy, math
import os

# matplotlib stuff
import matplotlib; matplotlib.use('pdf') #avoids TkInter threaded start
import matplotlib.pyplot as mpl
from matplotlib.backends.backend_pdf import PdfPages

# enable LaTeX interpreter
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
matplotlib.rc('lines', linewidth = 4)
# increase the default font size
matplotlib.rc('font', size=18)



def command_line_arguments(command_line_parameters):
  """Parse the program options"""

  # set up command line parser
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('-d', '--dev-files', required=True, nargs='+', help = "A list of score files of the development set.")
  parser.add_argument('-e', '--eval-files', nargs='+', help = "A list of score files of the evaluation set; if given it must be the same number of files as the --dev-files.")

  parser.add_argument('-s', '--directory', default = '.', help = "A directory, where to find the --dev-files and the --eval-files")

  parser.add_argument('-c', '--criterion', choices = ('EER', 'HTER'), help = "If given, the threshold of the development set will be computed with this criterion.")
  parser.add_argument('-l', '--legends', nargs='+', help = "A list of legend strings used for ROC, CMC and DET plots; if given, must be the same number than --dev-files.")
  parser.add_argument('-R', '--roc', help = "If given, ROC curves will be plotted into the given pdf file.")
  parser.add_argument('-D', '--det', help = "If given, DET curves will be plotted into the given pdf file.")
  parser.add_argument('-C', '--cmc', help = "If given, CMC curves will be plotted into the given pdf file.")
  parser.add_argument('-p', '--parser', default = '4column', choices = ('4column', '5column'), help="The style of the resulting score files. The default fits to the usual output of FaceRecLib score files.")

  parser.add_argument('--self-test', action='store_true', help=argparse.SUPPRESS)

  utils.add_logger_command_line_option(parser)

  # parse arguments
  args = parser.parse_args(command_line_parameters)

  utils.set_verbosity_level(args.verbose)

  # some sanity checks:
  if args.eval_files and len(args.dev_files) != len(args.eval_files):
    utils.error("The number of --dev-files (%d) and --eval-files (%d) are not identical" % (len(args.dev_files), len(args.eval_files)))

  if args.legends and len(args.dev_files) != len(args.legends):
    utils.error("The number of --dev-files (%d) and --legends (%d) are not identical" % (len(args.dev_diles), len(args.legends)))

  # assign the score file parser


  return args

def _plot_roc(frrs, colors, labels, title):
  figure = mpl.figure()
  # plot FAR and FRR for each algorithm
  for i in range(len(frrs)):
    mpl.semilogx([100.0*f for f in frrs[i][0]], [100.0*f for f in frrs[i][1]], color=colors[i], lw=2, ms=10, mew=1.5, label=labels[i])

  # finalize plot
  mpl.plot([0.1,0.1],[0,100], "--", color=(0.3,0.3,0.3))
  mpl.axis([frrs[0][0][0]*100,100,0,100])
  mpl.xticks((0.01, 0.1, 1, 10, 100), ('0.01', '0.1', '1', '10', '100'))
  mpl.xlabel('FAR (\%)')
  mpl.ylabel('CAR (\%)')
  mpl.grid(True, color=(0.6,0.6,0.6))
  mpl.legend(loc=4)
  mpl.title(title)

  return figure


def _plot_det(dets, colors, labels, title):
  # open new page for current plot
  figure = mpl.figure(figsize=(8.2,8))

  # plot the DET curves
  for i in range(len(dets)):
    mpl.plot(dets[i][0], dets[i][1], color=colors[i], lw=2, ms=10, mew=1.5, label=labels[i])

  # change axes accordingly
  det_list = [0.0002, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9, 0.95]
  ticks = [bob.measure.ppndf(d) for d in det_list]
  labels = [("%.5f" % (d*100)).rstrip('0').rstrip('.') for d in det_list]
  mpl.xticks(ticks, labels)
  mpl.yticks(ticks, labels)
  mpl.axis((ticks[0], ticks[-1], ticks[0], ticks[-1]))

  mpl.xlabel('FAR (\%)')
  mpl.ylabel('FRR (\%)')
  mpl.legend(loc=1)
  mpl.title(title)

  return figure

def _plot_cmc(cmcs, colors, labels, title):
  # open new page for current plot
  figure = mpl.figure()

  max_x = 0
  # plot the DET curves
  for i in range(len(cmcs)):
    x = bob.measure.plot.cmc(cmcs[i], figure=figure, color=colors[i], lw=2, ms=10, mew=1.5, label=labels[i])
    max_x = max(x, max_x)

  # change axes accordingly
  ticks = [int(t) for t in mpl.xticks()[0]]
  mpl.xlabel('Rank')
  mpl.ylabel('Probability (\%)')
  mpl.xticks(ticks, [str(t) for t in ticks])
  mpl.axis([0, max_x, 0, 100])
  mpl.legend(loc=4)
  mpl.title(title)

  return figure


def main(command_line_parameters=None):
  """Reads score files, computes error measures and plots curves."""

  args = command_line_arguments(command_line_parameters)

  # get some colors for plotting
  cmap = mpl.cm.get_cmap(name='hsv')
  colors = [cmap(i) for i in numpy.linspace(0, 1.0, len(args.dev_files)+1)]

  if args.criterion or args.roc or args.det:
    score_parser = {'4column' : bob.measure.load.split_four_column, '5column' : bob.measure.load.split_five_column}[args.parser]

    # First, read the score files
    utils.info("Loading %d score files of the development set" % len(args.dev_files))
    scores_dev = [score_parser(os.path.join(args.directory, f)) for f in args.dev_files]

    if args.eval_files:
      utils.info("Loading %d score files of the evaluation set" % len(args.eval_files))
      scores_eval = [score_parser(os.path.join(args.directory, f)) for f in args.eval_files]

    if args.criterion:
      utils.info("Computing %s on the development " % args.criterion + ("and HTER on the evaluation set" if args.eval_files else "set"))
      for i in range(len(scores_dev)):
        # compute threshold on development set
        threshold = {'EER': bob.measure.eer_threshold, 'HTER' : bob.measure.min_hter_threshold} [args.criterion](scores_dev[i][0], scores_dev[i][1])
        # apply threshold to development set
        far, frr = bob.measure.farfrr(scores_dev[i][0], scores_dev[i][1], threshold)
        print("The %s of the development set of '%s' is %2.3f%%" % (args.criterion, args.legends[i] if args.legends else args.dev_files[i], (far + frr) * 50.)) # / 2 * 100%
        if args.eval_files:
          # apply threshold to evaluation set
          far, frr = bob.measure.farfrr(scores_eval[i][0], scores_eval[i][1], threshold)
          print("The HTER of the evaluation set of '%s' is %2.3f%%" % (args.legends[i] if args.legends else args.dev_files[i], (far + frr) * 50.)) # / 2 * 100%

    if args.roc:
      utils.info("Computing CAR curves on the development " + ("and on the evaluation set" if args.eval_files else "set"))
      fars = [math.pow(10., i * 0.25) for i in range(-16,0)] + [1.]
      frrs_dev = [bob.measure.roc_for_far(scores[0], scores[1], fars) for scores in scores_dev]
      if args.eval_files:
        frrs_eval = [bob.measure.roc_for_far(scores[0], scores[1], fars) for scores in scores_eval]

      utils.info("Plotting ROC curves to file '%s'" % args.roc)
      # create a multi-page PDF for the ROC curve
      pdf = PdfPages(args.roc)
      # create a separate figure for dev and eval
      pdf.savefig(_plot_roc(frrs_dev, colors, args.legends if args.legends else args.dev_files, "ROC curve for development set"))
      del frrs_dev
      if args.eval_files:
        pdf.savefig(_plot_roc(frrs_eval, colors, args.legends if args.legends else args.eval_files, "ROC curve for evaluation set"))
        del frrs_eval
      pdf.close()


    if args.det:
      utils.info("Computing DET curves on the development " + ("and on the evaluation set" if args.eval_files else "set"))
      dets_dev = [bob.measure.det(scores[0], scores[1], 1000) for scores in scores_dev]
      if args.eval_files:
        dets_eval = [bob.measure.det(scores[0], scores[1], 1000) for scores in scores_eval]

      utils.info("Plotting DET curves to file '%s'" % args.det)
      # create a multi-page PDF for the ROC curve
      pdf = PdfPages(args.det)
      # create a separate figure for dev and eval
      pdf.savefig(_plot_det(dets_dev, colors, args.legends if args.legends else args.dev_files, "DET plot for development set"))
      del dets_dev
      if args.eval_files:
        pdf.savefig(_plot_det(dets_eval, colors, args.legends if args.legends else args.eval_files, "DET plot for evaluation set"))
        del dets_eval
      pdf.close()


  if args.cmc:
    utils.info("Computing CMC curves on the development " + ("and on the evaluation set" if args.eval_files else "set"))
    cmc_parser = {'4column' : bob.measure.load.cmc_four_column, '5column' : bob.measure.load.cmc_five_column}[args.parser]
    cmcs_dev = [cmc_parser(os.path.join(args.directory, f)) for f in args.dev_files]
    if args.eval_files:
      cmcs_eval = [cmc_parser(os.path.join(args.directory, f)) for f in args.eval_files]

    utils.info("Plotting CMC curves to file '%s'" % args.cmc)
    # create a multi-page PDF for the ROC curve
    pdf = PdfPages(args.cmc)
    # create a separate figure for dev and eval
    pdf.savefig(_plot_cmc(cmcs_dev, colors, args.legends if args.legends else args.dev_files, "CMC curve for development set"))
    if args.eval_files:
      pdf.savefig(_plot_cmc(cmcs_eval, colors, args.legends if args.legends else args.eval_files, "CMC curve for evaluation set"))
    pdf.close()



