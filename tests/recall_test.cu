/* Purpose: Check prepared recall order, scope, limits and exact recorded context.
 * Owns: Independent expected selections and failure controls.
 * Launch shape: Distinct N=1 and N=64 batches over real device kernels.
 * Lifetime: One test process; no language model is loaded. */
#include "recall_fixture.h"
#include "recall_state_cases.h"
#include <sstream>
#include <iomanip>

static std::string aotx_expected_row(uint64_t id, uint64_t version, unsigned source,
    unsigned evidence, unsigned reason, const std::string &text) {
    unsigned char bytes[16]; aotx_id(bytes, id); std::ostringstream s;
    s << "[memory id=" << std::hex << std::setfill('0');
    for (unsigned char b : bytes) s << std::setw(2) << (unsigned)b;
    s << std::dec << " version=" << version << " source=" << source << " evidence=" << evidence
      << " reason=" << reason << "]\n" << text << "\n"; return s.str();
}
static void aotx_ranking_cases(unsigned n) {
    aotx_recall_device d; auto f = aotx_memory_corpus(n, true); auto q = aotx_memory_queries(n, f.rows.size());
    aotx_check(!d.load(f.wire(false, f.rows.size())).status, "ranking corpus admission");
    auto rows = d.search(q, n); aotx_status_rows(rows, 0, "semantic search");
    const double vectors[3][3] = {{1,4,0},{2,1,0},{3,0,5}};
    for (unsigned i = 0; i < n; ++i) {
        float query[3]; memcpy(query, aotx_query_at(q,i) + 160, sizeof(query));
        double score[3]; std::array<unsigned,3> order = {0,1,2};
        for (unsigned j = 0; j < 3; ++j) {
            double dot=0,a=0,b=0; for (unsigned k=0;k<3;++k) {dot+=vectors[j][k]*query[k]; a+=vectors[j][k]*vectors[j][k];b+=(double)query[k]*query[k];}
            score[j]=dot/(std::sqrt(a)*std::sqrt(b));
        }
        std::sort(order.begin(),order.end(),[&](unsigned a,unsigned b){return score[a]>score[b];});
        aotx_check(rows[i].count == 3 && rows[i].searches == 1, "three semantic candidates");
        for(unsigned j=0;j<3;++j) aotx_check(aotx_selected(rows[i],j)==10000+i*3+order[j],"independent nonunit cosine order");
        auto p=aotx_query_at(q,i); aotx_pin(p,0,0,10002+i*3); aotx_pin(p,1,0,10000+i*3); aotx_pin(p,1,1,10002+i*3);
    }
    rows=d.search(q,n); aotx_status_rows(rows,0,"required and focus search");
    for(unsigned i=0;i<n;++i) {
        std::string expected;
        const unsigned order[3]={2,0,1}, reasons[3]={1,2,3};
        for(unsigned j=0;j<3;++j) {
            aotx_check(aotx_selected(rows[i],j)==10000+i*3+order[j] && rows[i].reason[j]==reasons[j],"required precedes higher cosine and deduplicates focus");
            expected+=aotx_expected_row(10000+i*3+order[j],1,3,0,reasons[j],"fact "+std::to_string(i)+" item "+std::to_string(order[j]));
        }
        aotx_put(aotx_query_at(q,i)+136,expected.size(),4);
        expected+="[input]\nrequest "+std::to_string(i);
        aotx_check(aotx_context(rows[i])==expected,"independent exact context bytes");
    }
    aotx_status_rows(d.search(q,n),0,"exact context budget fits");
    for(unsigned i=0;i<n;++i) {
        auto p=aotx_query_at(q,i); aotx_pin(p,0,1,10000+i*3); aotx_pin(p,0,2,10001+i*3);
        aotx_put(p+144,0,4); memset(p+4448,0,192); aotx_put(p+136,aotx_get(p+136,4)-1,4);
    }
    aotx_status_rows(d.search(q,n),AOTX_COG_CAPACITY,"required batch cannot lose one byte");
    q=aotx_memory_queries(n,f.rows.size());
    f.payloads[1]=f.payloads[0];
    aotx_check(!d.load(f.wire(false,f.rows.size())).status,"tie corpus admission"); rows=d.search(q,n);
    for(unsigned i=0;i<n;++i) {
        unsigned char a[16],b[16];aotx_id(a,10000+i*3);aotx_id(b,10001+i*3);
        aotx_check(aotx_selected(rows[i],0)==10000+i*3+(memcmp(a,b,16)<0?0:1),"exact ties use byte ID order");
    }
    f=aotx_memory_corpus(n,true);f.payloads[1][24]^=1;
    aotx_check(!d.load(f.wire(false,f.rows.size())).status,"mixed vector spaces");rows=d.search(q,n);
    for(unsigned i=0;i<n;++i)aotx_check(!rows[i].status&&rows[i].count==2&&
        aotx_selected(rows[i],0)==10000+i*3&&aotx_selected(rows[i],1)==10002+i*3,"compatible space filters unrelated vector index");
    f=aotx_memory_corpus(n,true);
    for(unsigned i=0;i<n;++i) {
        f.payloads[3+i*3]=aotx_memory_text(std::string(AOTX_RECALL_TEXT,'a'));
        f.payloads[4+i*3]=aotx_memory_text(std::string(AOTX_RECALL_TEXT,'b'));
        aotx_put(aotx_query_at(q,i)+136,200,4);aotx_put(aotx_query_at(q,i)+132,1,4);
    }
    aotx_check(!d.load(f.wire(false,f.rows.size())).status,"large optional context setup");rows=d.search(q,n);
    for(unsigned i=0;i<n;++i)aotx_check(!rows[i].status&&rows[i].count==1&&aotx_selected(rows[i],0)==10002+i*3,
        "oversized optional candidates leave room for smaller fact");
    for(unsigned i=0;i<n;++i)aotx_pin(aotx_query_at(q,i),0,0,10000+i*3);
    aotx_status_rows(d.search(q,n),2,"oversized required fact cannot be silently skipped");
}
static void aotx_query_cases(unsigned n) {
    aotx_recall_device d; auto f=aotx_memory_corpus(n); auto q=aotx_memory_queries(n,f.rows.size());
    aotx_check(!d.load(f.wire(false,f.rows.size())).status,"query corpus admission");
    for(unsigned mode=0;mode<10;++mode) {
        auto bad=q;
        for(unsigned i=0;i<n;++i) {
            auto p=aotx_query_at(bad,i);
            if(mode==0) p[64]^=1;
            if(mode==1) p[96]^=1;
            if(mode==2) aotx_float_put(p+160,std::numeric_limits<float>::quiet_NaN());
            if(mode==3) memset(p+160,0,12);
            if(mode==4) p[4640]=0;
            if(mode==5) p[8191]=1;
            if(mode==6) {aotx_pin(p,0,0,10000+i*3);aotx_id(p+16,7000+i);}
            if(mode==7) {aotx_pin(p,0,0,10000+i*3,2);}
            if(mode==8) {aotx_pin(p,0,0,10000+i*3);aotx_put(p+152,2,4);}
            if(mode==9) {aotx_put(p+128,4,4);}
        }
        const unsigned status[10]={7,7,8,8,1,1,11,10,4,7};
        aotx_status_rows(d.search(bad,n),status[mode],"malformed or unauthorized query");
    }
    auto bad=q; aotx_put(bad.data()+32,f.rows.size()-1); aotx_status_rows(d.search(bad,n),10,"stale cut");
    bad=q;bad[16]^=1;aotx_status_rows(d.search(bad,n),7,"wrong lineage");
    bad=q;bad[44]=1;aotx_status_rows(d.search(bad,n),1,"reserved envelope");
    for(unsigned mode=0;mode<4;++mode) {
        auto broken=f;
        for(unsigned i=0;i<n;++i) {
            if(mode==0) memset(broken.payloads[i].data()+128,0,12);
            if(mode==1) aotx_float_put(broken.payloads[i].data()+128,std::numeric_limits<float>::infinity());
            if(mode==2) broken.payloads[n+i][32]=0xc0;
            if(mode==3) broken.payloads[i][120]=1;
        }
        aotx_check(!d.load(broken.wire(false,broken.rows.size())).status,"opaque prepared payload admitted before recall validation");
        aotx_status_rows(d.search(q,n),mode==2?1:8,"prepared payload rejected by consumer");
    }
    auto tiny=f;
    for(unsigned i=0;i<n;++i) tiny.payloads[i]=aotx_memory_vector(std::numeric_limits<float>::denorm_min(),0,0);
    aotx_check(!d.load(tiny.wire(false,tiny.rows.size())).status,"small finite vector corpus");
    aotx_status_rows(d.search(q,n),0,"double norm preserves nonzero F32 subnormal");
    auto room=aotx_memory_corpus(n,false,1);q=aotx_memory_queries(n,room.rows.size(),1);
    aotx_check(!d.load(room.wire(false,room.rows.size())).status,"room corpus");
    for(unsigned i=0;i<n;++i)aotx_pin(aotx_query_at(q,i),0,0,10000+i*3);
    aotx_status_rows(d.search(q,n),0,"room scope has its own required facts");
    for(unsigned i=0;i<n;++i)aotx_query_at(q,i)[32]^=1;
    aotx_status_rows(d.search(q,n),11,"wrong room cannot read required fact");
}
static void aotx_record_cases(unsigned n) {
    aotx_bytes saved;std::vector<aotx_recall_result> original;
    {
        aotx_recall_device d;auto f=aotx_memory_corpus(n);auto q=aotx_memory_queries(n,f.rows.size());
        aotx_check(!d.load(f.wire(false,f.rows.size())).status,"record setup");auto before=d.checkpoint();
        auto bad=q;aotx_query_at(bad,n-1)[156]=1;d.search(bad,n);
        aotx_check(d.record(bad,n).status==1,"one invalid query refuses whole recording");
        aotx_check(d.checkpoint()==before,"bad query leaves exact state");
        if(n==64) {
            bad=q;memcpy(aotx_query_at(bad,n-1),aotx_query_at(bad,0),16);d.search(bad,n);
            aotx_check(d.record(bad,n).status==5,"duplicate request ID across batch");
        }
        bad=q;aotx_id(aotx_query_at(bad,n-1),10000);d.search(bad,n);
        aotx_check(d.record(bad,n).status==5,"existing object ID cannot be a new request");
        original=d.search(q,n);aotx_status_rows(original,0,"portable selection");
        auto r=d.record(q,n,true);aotx_check(!r.status&&r.applied==2*n&&r.sequence==4*n,"atomic ordered request and selection batch");
        saved=d.checkpoint();
    }
    aotx_recall_device restored;aotx_check(!restored.load(saved).status,"fresh device restores saved context");
    auto q=restored.saved_queries(n);auto rows=restored.search(q,n,true);aotx_status_rows(rows,0,"replay without query path");
    for(unsigned i=0;i<n;++i) {
        aotx_check(rows[i].searches==0 && rows[i].cut==original[i].cut,"replay retains cut and performs zero search");
        aotx_check(aotx_context(rows[i])==aotx_context(original[i])&&rows[i].count==original[i].count&&
            !memcmp(rows[i].selection,original[i].selection,AOTX_RECALL_SELECTION),"exact saved choice and context");
    }
    aotx_check(restored.checkpoint()==saved,"replay does not mutate store");
}

int main() {
    unsigned runs=0;
    for(unsigned n:{1u,64u}) {
        aotx_ranking_cases(n);aotx_query_cases(n);aotx_record_cases(n);
        aotx_recall_capacity_cases(n);aotx_recall_change_cases(n);
        aotx_recall_history_cases(n);aotx_recall_max_cut_cases(n);
        ++runs;printf("recall N=%u complete\n",n);
    }
    aotx_check(runs==2 && aotx_checks>1000,"both nonempty batch sizes ran");
    printf("recall: %u checks, %u failures\n",aotx_checks,aotx_failures);
    return aotx_failures?1:0;
}
