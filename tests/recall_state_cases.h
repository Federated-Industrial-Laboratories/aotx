/* Purpose: Check recall publication limits and current reference changes after restore.
 * Owns: Capacity and stale-selection fixtures independent of the scoring implementation.
 * Launch shape: Distinct N=1 and N=64 request and mutation batches.
 * Lifetime: One GPU test process. */
#ifndef AOTX_RECALL_STATE_CASES_H
#define AOTX_RECALL_STATE_CASES_H
#include "recall_fixture.h"

static void aotx_recall_capacity_cases(unsigned n) {
    aotx_recall_device d; aotx_fixture f;
    f.add(aotx_memory_row(0,AOTX_COG_COMPONENT,900000,1,2),aotx_memory_vector(3,2,1));
    for(unsigned i=0;i<n;++i) {
        auto r=aotx_memory_row(i,AOTX_COG_ASSERTION,10000+i*3,f.rows.size()+1);
        aotx_id(r.data()+AOTX_CO_EMBEDDING,900000);aotx_put(r.data()+AOTX_CO_EMBED_VERSION,1);
        f.add(r,aotx_memory_text("capacity "+std::to_string(i)));
    }
    size_t used=0;for(auto &p:f.payloads)used+=p.size();
    size_t retained=AOTX_COG_PAYLOAD-used-n*(AOTX_RECALL_QUERY+16+48);
    f.add(aotx_memory_row(0,AOTX_COG_COMPONENT,888888,f.rows.size()+1,2),aotx_bytes(retained+1,0x6d));
    auto q=aotx_memory_queries(n,f.rows.size());
    aotx_check(!d.load(f.wire(false,f.rows.size())).status,"aggregate memory capacity setup");
    auto before=d.checkpoint();aotx_status_rows(d.search(q,n),0,"capacity search still fits context");
    aotx_check(d.record(q,n).status==AOTX_COG_CAPACITY,"all recorded query bytes count towards capacity");
    aotx_check(d.checkpoint()==before,"aggregate capacity preserves exact live state");
    f.payloads.back().pop_back();aotx_check(!d.load(f.wire(false,f.rows.size())).status,"exact capacity setup");
    d.search(q,n);auto result=d.record(q,n,true);
    aotx_check(!result.status && result.applied==2*n,"aggregate exact fit records complete batch");
    auto exact=d.checkpoint();aotx_check(aotx_get(exact.data()+24)==AOTX_COG_PAYLOAD,"exact payload cap reached");
    auto full=aotx_memory_corpus(n);
    while(full.rows.size()<=AOTX_COG_OBJECTS-2*n)
        full.add(aotx_memory_row(0,AOTX_COG_COMPONENT,800000+full.rows.size(),full.rows.size()+1,2),{0x71});
    q=aotx_memory_queries(n,full.rows.size());aotx_check(!d.load(full.wire(false,full.rows.size())).status,"object capacity setup");
    before=d.checkpoint();d.search(q,n);aotx_check(d.record(q,n).status==2,"aggregate object capacity refusal");
    aotx_check(d.checkpoint()==before,"object capacity exact rollback");
}
static void aotx_recall_change_cases(unsigned n) {
    aotx_recall_device d; aotx_fixture f;
    for(unsigned i=0;i<n;++i)f.add(aotx_memory_row(i,AOTX_COG_ASSERTION,10000+i*3,i+1),aotx_memory_text("pending fact "+std::to_string(i)));
    auto q=aotx_memory_queries(n,n);
    for(unsigned i=0;i<n;++i)aotx_pin(aotx_query_at(q,i),0,0,10000+i*3);
    aotx_check(!d.load(f.wire(false,n)).status,"required only corpus");
    aotx_status_rows(d.search(q,n),0,"required facts need no vector component");
    aotx_check(!d.record(q,n,true).status,"record required-only selections");auto saved=d.checkpoint();
    for(unsigned mode=0;mode<4;++mode) {
        aotx_check(!d.load(saved).status,"restore before source revision"); aotx_fixture updates;
        for(unsigned i=0;i<n;++i) {
            auto r=f.rows[i];aotx_put(r.data()+AOTX_CO_VERSION,2);aotx_put(r.data()+AOTX_CO_UPDATED,3*n+i+1);
            auto p=aotx_memory_text("corrected fact "+std::to_string(i));
            if(mode==1) {aotx_put(r.data()+AOTX_CO_FLAGS,AOTX_COG_TOMBSTONE,4);p.clear();}
            if(mode==2) aotx_put(r.data()+AOTX_CO_EVIDENCE,3,4);
            if(mode==3) {
                aotx_id(r.data()+AOTX_CO_ID,600000+i);aotx_put(r.data()+AOTX_CO_VERSION,1);
                aotx_put(r.data()+AOTX_CO_CREATED,3*n+i+1);
                aotx_id(r.data()+AOTX_CO_SUPERSEDES,10000+i*3);aotx_put(r.data()+AOTX_CO_SUPER_VERSION,1);
            }
            updates.add(r,p);
        }
        aotx_check(!d.load(updates.wire(true,3*n+1,7),true).status,"admit recorded source change");
        auto queries=d.saved_queries(n);
        aotx_status_rows(d.search(queries,n,true),(mode==0||mode==3)?10:11,"saved selection refuses changed source");
    }
    auto expiring=f;
    for(auto &r:expiring.rows)aotx_put(r.data()+AOTX_CO_EXPIRY,2*n);
    aotx_check(!d.load(expiring.wire(false,n)).status,"expiry crossing setup");
    aotx_status_rows(d.search(q,n),0,"facts current at search cut");
    auto before=d.checkpoint();aotx_check(d.record(q,n).status==11,"publication rechecks future commit cut");
    aotx_check(d.checkpoint()==before,"expiry refusal preserves state");
    auto sources=aotx_memory_corpus(n);
    for(unsigned i=0;i<n;++i) {
        aotx_put(sources.rows[i].data()+AOTX_CO_EVIDENCE,3,4);
        aotx_id(sources.rows[n+i].data()+AOTX_CO_SOURCE,900000+i);aotx_put(sources.rows[n+i].data()+AOTX_CO_SOURCE_VERSION,1);
    }
    q=aotx_memory_queries(n,sources.rows.size());
    for(unsigned i=0;i<n;++i)aotx_pin(aotx_query_at(q,i),0,0,10000+i*3);
    aotx_check(!d.load(sources.wire(false,sources.rows.size())).status,"withdrawn dependency setup");
    aotx_status_rows(d.search(q,n),11,"withdrawn dependency prevents recall");
}
static aotx_bytes aotx_recall_history_queries(unsigned n, unsigned offset, uint64_t sequence) {
    auto q=aotx_memory_queries(n,sequence);
    for(unsigned i=0;i<n;++i) {
        auto p=aotx_query_at(q,i);
        aotx_id(p,100000+offset+i);aotx_id(p+48,200000+offset+i);
        aotx_pin(p,0,0,10000);
    }
    return q;
}
static void aotx_recall_history_cases(unsigned n) {
    aotx_recall_device d; aotx_fixture f;
    f.add(aotx_memory_row(0,AOTX_COG_ASSERTION,10000,1,2),aotx_memory_text("shared saved fact"));
    for(unsigned total:{64u,65u}) {
        aotx_check(!d.load(f.wire(false,1)).status,"saved history setup");
        unsigned first=total-n;
        if(first) {
            auto q=aotx_recall_history_queries(first,0,1);
            aotx_status_rows(d.search(q,first),0,"first saved history search");
            aotx_check(!d.record(q,first,true).status,"first saved history record");
        }
        auto before=d.checkpoint();auto q=aotx_recall_history_queries(n,first,1+2*first);
        aotx_status_rows(d.search(q,n),0,"later batch still fits search");
        auto result=d.record(q,n,true);
        if(total==65) {
            aotx_check(result.status==AOTX_COG_CAPACITY,"saved query limit includes earlier batches");
            aotx_check(d.checkpoint()==before,"saved query overflow preserves state");
        } else aotx_check(!result.status&&result.applied==2*n,"exact saved query limit records all rows");
        unsigned saved=total==65?first:total;
        auto checkpoint=d.checkpoint();aotx_check(!d.load(checkpoint).status,"restore accumulated history");
        auto requests=d.saved_queries(saved);auto rows=d.search(requests,saved,true);
        aotx_status_rows(rows,0,"all admitted saved history replays");
        for(unsigned i=0;i<saved;++i) aotx_check(rows[i].searches==0&&aotx_selected(rows[i],0)==10000,
            "accumulated history keeps exact choice without search");
        aotx_check(d.checkpoint()==checkpoint,"accumulated replay keeps state");
    }
}
static void aotx_recall_max_cut_cases(unsigned n) {
    aotx_recall_device d;uint64_t cut=UINT64_MAX-2*n;
    for(unsigned expires:{0u,1u,2u}) {
        aotx_fixture f;
        f.add(aotx_memory_row(0,AOTX_COG_COMPONENT,900000,1,2),aotx_memory_vector(2,3,1));
        if(expires==2)aotx_put(f.rows[0].data()+AOTX_CO_EXPIRY,UINT64_MAX);
        for(unsigned i=0;i<n;++i) {
            auto r=aotx_memory_row(i,AOTX_COG_ASSERTION,10000+i*3,i+2);
            aotx_id(r.data()+AOTX_CO_EMBEDDING,900000);aotx_put(r.data()+AOTX_CO_EMBED_VERSION,1);
            if(expires==1)aotx_put(r.data()+AOTX_CO_EXPIRY,UINT64_MAX);
            f.add(r,aotx_memory_text("last cut fact "+std::to_string(i)));
        }
        auto q=aotx_memory_queries(n,cut);
        for(unsigned i=0;i<n;++i)aotx_pin(aotx_query_at(q,i),0,0,10000+i*3);
        aotx_check(!d.load(f.wire(false,cut)).status,"maximum publication cut setup");
        aotx_status_rows(d.search(q,n),0,"near-maximum search cut is valid");
        auto before=d.checkpoint();auto result=d.record(q,n,true);
        if(expires) {
            aotx_check(result.status==AOTX_COG_DENIED,"maximum explicit cut enforces direct and source expiry");
            aotx_check(d.checkpoint()==before,"maximum-cut expiry refuses complete batch");
        } else {
            aotx_check(!result.status&&result.sequence==UINT64_MAX,"maximum cut remains valid without expiry");
            auto saved=d.checkpoint();aotx_check(!d.load(saved).status,"restore maximum publication cut");
            auto requests=d.saved_queries(n);auto rows=d.search(requests,n,true);
            aotx_status_rows(rows,0,"maximum-cut control replays");
            for(auto &r:rows)aotx_check(r.cut==cut&&!r.searches,"maximum-cut original sequence preserved");
        }
    }
}
#endif
