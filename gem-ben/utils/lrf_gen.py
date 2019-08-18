# -*- coding: utf-8 -*-
from time import time
from argparse import ArgumentParser
import numpy as np


def integral(a,b):

    if (abs(a + 1.) > 1e-10):
        intg_val = (1. / (a + 1.) * pow(b, a + 1.))
    else:
        intg_val=  (np.log(b))

    return intg_val

def average_degree(dmax, dmin, gamma):
    return (1./(integral(gamma, dmax)-integral(gamma, dmin)))*(integral(gamma+1, dmax)-integral(gamma+1, dmin));


def solve_dmin(dmax, dmed, gamma):
    dmin_l = 1;
    dmin_r = dmax;
    average_k1 = average_degree(dmin_r, dmin_l, gamma);
    average_k2 = dmin_r;

    if ((average_k1 - dmed > 0) or (average_k2 - dmed < 0)):

        if (average_k1-dmed > 0):
            raise ("\nyou should increase the average degree (bigger than " +average_k1+"(or decrease the maximum degree...)")

        if (average_k2-dmed > 0):
            raise("\nyou should decrease the average degree (smaller than " +average_k2 +")(or increase the maximum degree...)")

        return -1;

    while (abs(average_k1 - dmed) > 1e-7):

        temp = average_degree(dmax, ((dmin_r+dmin_l) / 2.), gamma)

        if ((temp-dmed) * (average_k2-dmed) > 0):
            average_k2=temp;
            dmin_r=((dmin_r+dmin_l) / 2.);

        else:
            average_k1=temp;
            dmin_l=((dmin_r+dmin_l) / 2.);

    return dmin_l;

def benchmark(excess, defect, num_nodes, average_k, max_degree, tau, tau2, mixing_parameter,
                  overlapping_nodes, overlap_membership, nmin, nmax, fixed_range, clustering_coeff):
    dmin = solve_dmin(max_degree, average_k, -1*t1)
    print dmin
    exit()

if __name__ == '__main__':

    t1 = time()
    parser = ArgumentParser(description='Lancichinetti–Fortunato–Radicchi')


    parser.add_argument('-N',
                        help='number of nodes', default=1000)
    parser.add_argument('-k',
                        help='average degree',default=15)
    parser.add_argument('--maxk',
                        help='maximum degree', default=50)
    parser.add_argument('--mu',
                        help='mixing parameter',default=0.3)
    parser.add_argument('--t1',
                        help='minus exponent for the degree sequence',default=2)
    parser.add_argument('--t2',
                        help='minus exponent for the community size distribution', default=1)
    parser.add_argument('--minc',
                        help='minimum for the community sizes', default=20)
    parser.add_argument('--maxc',
                        help='maximum for the community sizes', default=50)
    parser.add_argument('--on',
                        help='number of overlapping nodes', default=0)
    parser.add_argument('--om',
                    help='number of memberships of the oberlapping nodes',default=0)
    parser.add_argument('--C',
                        help='Average clustering coefficient', default=-214741)
    parser.add_argument('--rand',
                        help='randomf', default='True')
    parser.add_argument('--excess',
                        help='excess', default='True')
    parser.add_argument('--defect',
                        help='defect', default='True')
    parser.add_argument('--fixed_range',
                        help='fixed range', default='False')


    params = {}

    args = vars(parser.parse_args())

    for k, v in args.items():
        if v is not None:
            params[k] = v

    benchmark(params['excess'], params['defect'], params['N'], params['k'], params['maxk'], params['t1'], params['t2'], params['mu'],
                  params['on'], params['om'], params['minc'], params['maxc'], params['fixed_range'], params['C'])
