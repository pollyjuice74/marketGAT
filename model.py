class MarketNN(nn.Module):
        def __init__(self):
                self.mask = mask
                
        def tran_call(self, graph):
            # (s, p)->(s, p/m + f/v, k+u, d)->(s, k+u)->(s, u)->(s,1)->(1,)  
            emb_symbols = [ emb_sym(sym_graph.x) for sym_graph in graph ] 


def tran_call(self, r_t, t):
        # Make sure r_t and t are compatible
        r_t = tf.reshape(r_t, (self.n, -1)) # (n,b)
        t = tf.cast(t, dtype=tf.int32)

        # Compute synd and magn
        syndrome = tf.reshape( self.get_syndrome(llr_to_bin(r_t)), (self.pcm.shape[0], -1) ) # (m,n)@(n,b)->(m,b) check nodes
        magnitude = tf.reshape( tf.abs(r_t), (self.n, -1) ) #(n,b) variable nodes
        # make sure their the same dtype
        magnitude, syndrome = [ tf.cast(tensor, dtype=tf.float32) for tensor in [magnitude, syndrome] ]

        # Concatenate synd and magn
        nodes = tf.concat([magnitude, syndrome], axis=0) # data for vertices
        nodes = tf.reshape(nodes, (1, self.n+self.m, -1)) # (1, n+m, b)
        # print(nodes.shape)

        # Embedding nodes w/ attn and 'time' (sum syn errs) dims
        nodes_emb = tf.reshape( self.src_embed * nodes, (self.src_embed.shape[-1], self.pcm.shape[0]+self.n, -1) ) # (d,n+m,b)
        time_emb = tf.reshape( self.time_embed(t), (self.src_embed.shape[-1], 1, -1) ) # (d,1,b)

        # Applying embeds
        emb_t = time_emb * nodes_emb # (d, n+m, b)
        logits = self.decoder(emb_t) # (d, n+m, d) # TODO: missing batch dims b
        # print(emb_t, logits)

        # Reduce (d,n+m,d)->(d,n+m)
        logits = tf.squeeze( self.fc(logits), axis=-1 )
        vn_logits = tf.reshape( logits[:, :self.n], (self.n, -1) ) # (n,d) take the first n logits from the concatenation
        cn_logits = tf.reshape( logits[:, self.n:], (self.m, -1) ) # (m,d) take the last m logits from the concatenation
        # print(vn_logits, cn_logits)

        z_hat = self.to_n(vn_logits)# (n,d)->(n,)
        synd = self.to_m(cn_logits)# (m,d)->(m,)
        # print(logits.shape, z_hat.shape)

        return z_hat, synd


def create_mask(self, H):
        m,n = H.shape
        mask = tf.eye(n+m, dtype=tf.float32) # (n+m, n+m)
        cn_con, vn_con, _ = sp.sparse.find(H)
        
        for cn, vn_i in zip(cn_con, vn_con):
            # cn to vn connections in the mask
            mask = tf.tensor_scatter_nd_update(mask, [[n+cn, vn_i],[vn_i, n+cn]], [1.0,1.0])
        
            # distance 2 vn neighbors of vn_i
            related_vns = vn_con[cn_con==cn]
            for vn_j in related_vns:
                mask = tf.tensor_scatter_nd_update(mask, [[vn_i, vn_j],[vn_j, vn_i]], [1.0,1.0])
        
        # -infinity where mask is not set
        mask = tf.cast( tf.math.logical_not(mask > 0), dtype=tf.float32) # not(mask > 0) for setting non connections to -1e9
        return mask
