import tensorflow as tf

def hamm(n, base):
    seq = tf.zeros((n,), dtype=tf.float32)

    if base > 1:
        seed = tf.range(1, n+1, dtype=tf.float32)
        base_inv = 1.0 / base
        while tf.reduce_any(seed != 0):
            digit = tf.math.floormod(seed, base)
            seq += digit * base_inv
            base_inv /= base
            seed = tf.floor(seed / base)
    else:
        temp = tf.range(1, n+1, dtype=tf.float32)
        seq = (tf.math.floormod(temp, (-base + 1)) + 0.5) / (-base)

    return seq


def zero_crossings(x):
    # Find zero crossings (where the sign changes between consecutive elements)
    indzer = tf.where(x[:-1] * x[1:] < 0)[:, 0]

    # Check if there are any exact zeros in the array
    if tf.reduce_any(x == 0):
        iz = tf.where(x == 0)[:, 0]

        # Check if consecutive zeros exist
        if tf.reduce_any(tf.diff(iz) == 1):
            zer = tf.cast(x == 0, tf.int32)
            dz = tf.diff(tf.concat([[0], zer, [0]], axis=0))

            debz = tf.where(dz == 1)[:, 0]
            finz = tf.where(dz == -1)[:, 0] - 1
            indz = tf.cast(tf.round((debz + finz) / 2), tf.int32)
        else:
            indz = iz

        # Combine the zero crossings and zero indices and sort them
        indzer = tf.sort(tf.concat([indzer, indz], axis=0))

    return indzer

def boundary_conditions(indmin, indmax, t, x, z, nbsym):
    lx = len(x) - 1
    end_max = len(indmax) - 1
    end_min = len(indmin) - 1
    indmin = tf.cast(indmin, tf.int32)
    indmax = tf.cast(indmax, tf.int32)

    if len(indmin) + len(indmax) < 3:
        mode = 0
        return None, None, None, None, mode
    else:
        mode = 1  # the projected signal has inadequate extrema

    # Boundary conditions for interpolations:
    if indmax[0] < indmin[0]:
        if x[0] > x[indmin[0]]:
            lmax = tf.reverse(indmax[1:min(end_max + 1, nbsym + 1)], axis=[0])
            lmin = tf.reverse(indmin[:min(end_min + 1, nbsym)], axis=[0])
            lsym = indmax[0]
        else:
            lmax = tf.reverse(indmax[:min(end_max + 1, nbsym)], axis=[0])
            lmin = tf.concat([tf.reverse(indmin[:min(end_min + 1, nbsym - 1)], axis=[0]), tf.constant([0], dtype=tf.int32)], axis=0)
            lsym = 0
    else:
        if x[0] < x[indmax[0]]:
            lmax = tf.reverse(indmax[:min(end_max + 1, nbsym)], axis=[0])
            lmin = tf.reverse(indmin[1:min(end_min + 1, nbsym + 1)], axis=[0])
            lsym = indmin[0]
        else:
            lmax = tf.concat([tf.reverse(indmax[:min(end_max + 1, nbsym - 1)], axis=[0]), tf.constant([0], dtype=tf.int32)], axis=0)
            lmin = tf.reverse(indmin[:min(end_min + 1, nbsym)], axis=[0])
            lsym = 0

    if indmax[-1] < indmin[-1]:
        if x[-1] < x[indmax[-1]]:
            rmax = tf.reverse(indmax[max(end_max - nbsym + 1, 0):], axis=[0])
            rmin = tf.reverse(indmin[max(end_min - nbsym, 0):-1], axis=[0])
            rsym = indmin[-1]
        else:
            rmax = tf.concat([tf.constant([lx], dtype=tf.int32), tf.reverse(indmax[max(end_max - nbsym + 2, 0):], axis=[0])], axis=0)
            rmin = tf.reverse(indmin[max(end_min - nbsym + 1, 0):], axis=[0])
            rsym = lx
    else:
        if x[-1] > x[indmin[-1]]:
            rmax = tf.reverse(indmax[max(end_max - nbsym, 0):-1], axis=[0])
            rmin = tf.reverse(indmin[max(end_min - nbsym + 1, 0):], axis=[0])
            rsym = indmax[-1]
        else:
            rmax = tf.reverse(indmax[max(end_max - nbsym + 1, 0):], axis=[0])
            rmin = tf.concat([tf.constant([lx], dtype=tf.int32), tf.reverse(indmin[max(end_min - nbsym + 2, 0):], axis=[0])], axis=0)
            rsym = lx

    tlmin = 2 * t[lsym] - t[lmin]
    tlmax = 2 * t[lsym] - t[lmax]
    trmin = 2 * t[rsym] - t[rmin]
    trmax = 2 * t[rsym] - t[rmax]

    # In case symmetrized parts do not extend enough
    if tlmin[0] > t[0] or tlmax[0] > t[0]:
        if lsym == indmax[0]:
            lmax = tf.reverse(indmax[:min(end_max + 1, nbsym)], axis=[0])
        else:
            lmin = tf.reverse(indmin[:min(end_min + 1, nbsym)], axis=[0])
        lsym = 0
        tlmin = 2 * t[lsym] - t[lmin]
        tlmax = 2 * t[lsym] - t[lmax]

    if trmin[-1] < t[lx] or trmax[-1] < t[lx]:
        if rsym == indmax[-1]:
            rmax = tf.reverse(indmax[max(end_max - nbsym + 1, 0):], axis=[0])
        else:
            rmin = tf.reverse(indmin[max(end_min - nbsym + 1, 0):], axis=[0])
        rsym = lx
        trmin = 2 * t[rsym] - t[rmin]
        trmax = 2 * t[rsym] - t[rmax]

    zlmax = tf.gather(z, lmax)
    zlmin = tf.gather(z, lmin)
    zrmax = tf.gather(z, rmax)
    zrmin = tf.gather(z, rmin)

    tmin = tf.concat([tlmin, tf.gather(t, indmin), trmin], axis=0)
    tmax = tf.concat([tlmax, tf.gather(t, indmax), trmax], axis=0)
    zmin = tf.concat([zlmin, tf.gather(z, indmin), zrmin], axis=0)
    zmax = tf.concat([zlmax, tf.gather(z, indmax), zrmax], axis=0)

    return tmin, tmax, zmin, zmax, mode


def envelope_mean(m, t, seq, ndir, N, N_dim):
    NBSYM = 2
    count = 0

    # Initialize arrays using TensorFlow
    env_mean = tf.zeros((len(t), N_dim), dtype=tf.float32)
    amp = tf.zeros(len(t), dtype=tf.float32)
    nem = tf.zeros(ndir, dtype=tf.float32)
    nzm = tf.zeros(ndir, dtype=tf.float32)

    dir_vec = tf.zeros((N_dim, 1), dtype=tf.float32)

    for it in range(ndir):
        if N_dim != 3:  # Multivariate signal
            # Linear normalization of hammersley sequence in the range -1.00 to 1.00
            b = 2 * seq[it, :] - 1

            # Find angles corresponding to the normalized sequence
            tht = tf.atan2(tf.sqrt(tf.cumsum(b[::-1][1:]**2)), b[:N_dim - 1])

            # Compute unit direction vectors on the n-sphere
            dir_vec = tf.concat([tf.ones(1), tf.sin(tht)], axis=0)
            dir_vec = tf.math.cumprod(dir_vec, axis=0)
            dir_vec[:N_dim - 1] = tf.cos(tht) * dir_vec[:N_dim - 1]

        else:  # Trivariate signal
            tt = 2 * seq[it, 0] - 1
            tt = tf.clip_by_value(tt, -1, 1)  # Clipping to ensure the range is [-1, 1]
            phirad = seq[it, 1] * 2 * np.pi
            st = tf.sqrt(1.0 - tt**2)

            # Compute direction vector
            dir_vec = tf.stack([st * tf.cos(phirad), st * tf.sin(phirad), tt])

        # Project input signal onto direction vectors
        y = tf.linalg.matvec(m, dir_vec)

        # Compute extrema of projected signal
        indmin, indmax = local_peaks(y.numpy())  # Assuming local_peaks uses numpy, not TensorFlow

        nem = nem + tf.cast(len(indmin) + len(indmax), tf.float32)
        indzer = zero_crossings(y)

        nzm[it] = tf.cast(len(indzer), tf.float32)

        # Boundary conditions for interpolation
        tmin, tmax, zmin, zmax, mode = boundary_conditions(indmin, indmax, t, y, m, NBSYM)

        # Perform cubic spline interpolation if the mode is valid
        if mode:
            # Use scipy CubicSpline for now (since TensorFlow doesn't directly support it)
            fmin = CubicSpline(tmin, zmin, bc_type='not-a-knot')
            env_min = fmin(t)
            fmax = CubicSpline(tmax, zmax, bc_type='not-a-knot')
            env_max = fmax(t)

            # Compute amplitude and mean of envelopes
            amp += tf.sqrt(tf.reduce_sum(tf.square(env_max - env_min), axis=1)) / 2
            env_mean += (env_max + env_min) / 2
        else:
            count += 1

    # Normalize the envelope mean and amplitude
    if ndir > count:
        env_mean = env_mean / (ndir - count)
        amp = amp / (ndir - count)
    else:
        env_mean = tf.zeros((N, N_dim), dtype=tf.float32)
        amp = tf.zeros(N, dtype=tf.float32)
        nem = tf.zeros(ndir, dtype=tf.float32)

    return env_mean, nem, nzm, amp