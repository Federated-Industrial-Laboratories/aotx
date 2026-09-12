/* Purpose: Check portable image budgets and their allocation bounds.
 * Owns: Independent profile byte fixtures and malformed capacity fields.
 * Threading: One disk test over single and 64-profile batches.
 * Lifetime: One test process. */
#include "disk/modelfile/media_profile.h"
#include "disk/modelfile/vision.h"
#include "disk/modelfile/manifest.h"
#include "cuda/media/wire.h"
#include <stdio.h>
#include <string.h>

static unsigned checks, failures;
static void check(int good, const char *name) {
    ++checks; if (!good) { ++failures; fprintf(stderr, "FAIL %s\n", name); }
}
static int equal(const aotx_media_profile *a, const aotx_media_profile *b) {
    return a->objects == b->objects && a->bytes == b->bytes && a->feature_rows == b->feature_rows &&
        a->workers == b->workers && a->pixels == b->pixels && a->dimension == b->dimension &&
        a->patches == b->patches && a->horizontal == b->horizontal;
}
static void profiles(unsigned count) {
    for (unsigned i = 0; i < count; ++i) {
        aotx_media_profile a, b; unsigned char raw[80], bad[80];
        aotx_media_profile_default(&a);
        a.objects += i; a.bytes += 31u*i; a.feature_rows += i;
        aotx_media_profile_write(&a, raw);
        check(!memcmp(raw, "AOTXIM01", 8) && aotx_media_get(raw+8,4) == 1 &&
            aotx_media_get(raw+12,4) == a.objects && aotx_media_get(raw+16,8) == a.bytes &&
            aotx_media_get(raw+48,8) == a.horizontal, "portable fields use declared little-endian offsets");
        check(!aotx_media_profile_read(raw, sizeof raw, &b) && equal(&a, &b), "distinct profile survives encoding");
        if (!i) check(aotx_media_profile_fits(&a), "compiled default fits");
        else check(!aotx_media_profile_fits(&a), "larger resident requirements do not fit this build");
        for (unsigned j = 0; j < sizeof raw; ++j) {
            if (!(j < 8 || (j >= 44 && j < 48) || j >= 56)) continue;
            memcpy(bad, raw, sizeof bad); bad[j] ^= 1;
            memset(&b, 0xa5, sizeof b); aotx_media_profile before = b;
            check(aotx_media_profile_read(bad, sizeof bad, &b) && !memcmp(&before,&b,sizeof b),
                  "invalid magic or reserved data changes no output field");
        }
        const unsigned offsets[] = {8,12,16,24,28,32,36,40,48};
        for (unsigned j = 0; j < sizeof offsets / sizeof offsets[0]; ++j) {
            memcpy(bad,raw,sizeof bad); aotx_media_put(bad+offsets[j],0,offsets[j] == 16 || offsets[j] == 48 ? 8 : 4);
            check(aotx_media_profile_read(bad,sizeof bad,&b), "zero required capacity or schema is refused");
        }
        check(aotx_media_profile_read(raw,79,&b) && aotx_media_profile_read(raw,81,&b), "profile length is exact");
        memcpy(bad,raw,sizeof bad); aotx_media_put(bad+16,UINT64_MAX,8);
        check(aotx_media_profile_read(bad,sizeof bad,&b), "source digest length overflow is refused");
        memcpy(bad,raw,sizeof bad); aotx_media_put(bad+48,UINT64_MAX,8);
        check(aotx_media_profile_read(bad,sizeof bad,&b), "resize allocation overflow is refused");
        memcpy(bad,raw,sizeof bad); aotx_media_put(bad+40,257,4);
        check(aotx_media_profile_read(bad,sizeof bad,&b), "partial merge groups are refused");
        memcpy(bad,raw,sizeof bad); aotx_media_put(bad+40,65540,4);
        check(aotx_media_profile_read(bad,sizeof bad,&b), "trained pixel extent is enforced");
        memcpy(bad,raw,sizeof bad); aotx_media_put(bad+36,65536,4);
        check(aotx_media_profile_read(bad,sizeof bad,&b), "codec dimension overflow is refused");
    }
}
static int manifest(const aotx_manifest_entry pair[2], aotx_manifest_entry out[2]) {
    char text[2*AOTX_MANIFEST_LINE], line[AOTX_MANIFEST_LINE];
    if (aotx_manifest_write_line(text,sizeof text,pair) ||
        aotx_manifest_write_line(line,sizeof line,pair+1)) return -2;
    strcat(text,line);return aotx_vision_manifest_text(text,strlen(text),out);
}
static void pairs(unsigned count) {
    for (unsigned i=0;i<count;++i) {
        aotx_manifest_entry pair[2]={{0}},out[2];
        for(unsigned j=0;j<2;++j) {
            aotx_manifest_entry *p=pair+j;
            snprintf(p->name,sizeof p->name,"component-%u-%u",i,j);
            snprintf(p->path,sizeof p->path,"weights/%u-%u.gguf",i,j);
            strcpy(p->role,j ? "vision" : "language");strcpy(p->source,"source/model");
            strcpy(p->revision,"0123456789abcdef");strcpy(p->license,"Apache-2.0");
            memset(p->sha256,j ? 'b' : 'a',64);p->sha256[64]=0;p->bytes=1000+i+j;
        }
        check(manifest(pair,out)==2 && aotx_vision_pair(out,pair,1)==0,"distinct paired manifests bind their exact parent");
        aotx_manifest_entry models[2]={pair[0],pair[0]};
        check(aotx_vision_pair(pair,models,2)<0,"duplicate parent identities are refused");
        for(unsigned mode=0;mode<10;++mode) {
            aotx_manifest_entry changed[2]={pair[0],pair[1]};
            if(mode==0) strcpy(changed[1].role,"language");
            if(mode==1) strcpy(changed[0].role,"embedding");
            if(mode==2) strcpy(changed[1].path,changed[0].path);
            if(mode==3) strcpy(changed[1].name,changed[0].name);
            if(mode==4) strcpy(changed[1].sha256,changed[0].sha256);
            if(mode==5) changed[1].bytes=0;
            if(mode==6) changed[1].revision[0]=0;
            if(mode==7) changed[1].license[0]=0;
            if(mode==8) changed[1].source[0]=0;
            if(mode==9) changed[0].revision[0]=0;
            check(manifest(changed,out)<0,"invalid paired roles, identity and provenance are refused");
        }
        for(unsigned mode=0;mode<8;++mode) {
            aotx_manifest_entry parent=pair[0];
            if(mode==0) parent.bytes++;
            if(mode==1) parent.sha256[0]='c';
            if(mode==2) parent.name[0]='z';
            if(mode==3) parent.path[0]='z';
            if(mode==4) parent.source[0]='z';
            if(mode==5) parent.revision[0]='z';
            if(mode==6) parent.license[0]='z';
            if(mode==7) strcpy(parent.role,"language-q4");
            check(aotx_vision_pair(pair,&parent,1)<0,"every parent identity field binds the component");
        }
        check(aotx_vision_manifest_text("",0,out)<0 &&
            aotx_vision_manifest_text("x\0y",3,out)<0,"empty and embedded-null manifests are refused");
    }
}
int main(void) {
    profiles(1); profiles(64);pairs(1);pairs(64);
    printf("checks=%u failures=%u\n",checks,failures); return failures ? 1 : 0;
}
